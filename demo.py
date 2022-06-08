import argparse

import cv2
import numpy as np
import torch
import threading
import tkinter
import pyrealsense2 as rs

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose
from val import normalize, pad_width


class VideoReader(object):
    def __init__(self, file_name):
        self.file_name = file_name
        try:  # OpenCV needs int to read from webcam
            self.file_name = int(file_name)
        except ValueError:
            pass

    def __iter__(self):
        self.cap = cv2.VideoCapture(self.file_name)
        if not self.cap.isOpened():
            raise IOError('Video {} cannot be opened'.format(self.file_name))
        return self

    def __next__(self):
        was_read, img = self.cap.read()
        if not was_read:
            raise StopIteration
        return img

class PullupArgs(object):
    def __init__(self, bar_height, bar_width, toplimit_center, toplimit_width,
                 lowerlimit_offset, lowerlimit_width, baseline_offset,
                 if_draw_line=1, if_draw_state=1):
        self.bar_height = bar_height
        self.bar_width = bar_width
        self.toplimit_center = toplimit_center
        self.toplimit_width = toplimit_width
        self.lowerlimit_offset = lowerlimit_offset
        self.lowerlimit_center = 0
        self.lowerlimit_width = lowerlimit_width
        self.baseline_offset = baseline_offset
        self.baseline = 0

        self.if_draw_line = if_draw_line
        self.if_draw_state = if_draw_state

        self.pullup_state = 0
        self.pullup_cnt = 0


def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu,
               pad_value=(0, 0, 0), img_mean=np.array([128, 128, 128], np.float32), img_scale=np.float32(1/256)):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    if not cpu:
        tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def human_pose_2d(net, img, height_size, cpu):
    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    
    heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

    total_keypoints_num = 0
    all_keypoints_by_type = []
    for kpt_idx in range(num_keypoints):  # 19th for bg
        total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
        all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)

    return current_poses

def demo_images(net, images, height_size, cpu):
    img_paths = images
    for img_path in img_paths:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_img = img.copy()
        current_poses = human_pose_2d(net, img, height_size, cpu)

        # for idx, pose in enumerate(current_poses):
        #     print("id {}:".format(idx))
        #     print(pose.keypoints)
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(0)

def demo_video(net, video, height_size, cpu):
    frame_provider = VideoReader(video)
    delay = 1
    for img in frame_provider:
        orig_img = img.copy()
        current_poses = human_pose_2d(net, img, height_size, cpu)

        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            break
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


def gui_pullup_args(pullup_args):
    def func_scale_bar_height(v):
        pullup_args.bar_height = int(v)
    def func_scale_bar_width(v):
        pullup_args.bar_width = int(v)
    def func_scale_toplimit_center(v):
        pullup_args.toplimit_center = int(v)
    def func_scale_toplimit_width(v):
        pullup_args.toplimit_width = int(v)
    def func_scale_baseline_offset(v):
        pullup_args.baseline_offset = int(v)
    def func_scale_lowerlimt_offset(v):
        pullup_args.lowerlimt_offset = int(v)
    def func_scale_lowerlimt_width(v):
        pullup_args.lowerlimt_width = int(v)
    def func_button_if_draw_line():
        if pullup_args.if_draw_line:
            pullup_args.if_draw_line = 0
            button_if_draw_line['text'] = "显示测试曲线"
        else:
            pullup_args.if_draw_line = 1
            button_if_draw_line['text'] = "不显示测试曲线"
    def func_button_if_draw_state():
        if pullup_args.if_draw_state:
            pullup_args.if_draw_state = 0
            button_if_draw_state['text'] = "显示测试信息"
        else:
            pullup_args.if_draw_state = 1
            button_if_draw_state['text'] = "不显示测试信息"
 
    window = tkinter.Tk()
    window.title("GUI for pullup_args")
    window.geometry("1280x580")

    label_explanation0 = tkinter.Label(text = "参数说明：")
    label_explanation0.place(x=650, y=41)
    label_explanation1 = tkinter.Label(text = "单杠高度为bar_height，上杠条件为手部高度在[bar_height-bar_width, bar_height+bar_width]范围内")
    label_explanation1.place(x=650, y=111)
    label_explanation2 = tkinter.Label(text = "肩部的初始高度为baseline+baseline_offset，下杠判断为肩膀高度 > baseline")
    label_explanation2.place(x=650, y=181)
    label_explanation3 = tkinter.Label(text = "完成一个上升的标志为肩膀高度进入[toplimit_center-toplimit_width, toplimit_center_toplimit_width]区间")
    label_explanation3.place(x=650, y=251)
    label_explanation4 = tkinter.Label(text = "完成一个下降的标志为肩膀高度进入[lowerlimt_center-lowerlimt_width, lowerlimt_center+lowerlimt_width]区间")
    label_explanation4.place(x=650, y=321)
    label_explanation5 = tkinter.Label(text = "其中lowerlimt_center = baseline - lowerlimt_offset")
    label_explanation5.place(x=650, y=391)

    button_if_draw_line = tkinter.Button(window, text="不显示测试曲线", command=func_button_if_draw_line)
    button_if_draw_line.place(x=650, y=458)
    button_if_draw_state = tkinter.Button(window, text="不显示测试信息", command=func_button_if_draw_state)
    button_if_draw_state.place(x=850, y=458)

    label_bar_height = tkinter.Label(text = "bar_height")
    label_bar_height.place(x=20, y=41)
    scale_bar_height = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, 
                                     command=func_scale_bar_height)
    scale_bar_height.place(x=130, y=20)
    scale_bar_height.set(value=130)

    label_bar_width = tkinter.Label(text = "bar_width")
    label_bar_width.place(x=20, y=111)
    scale_bar_width = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500,
                                    command=func_scale_bar_width)
    scale_bar_width.place(x=130, y=90)
    scale_bar_width.set(value=40)

    label_toplimit_center = tkinter.Label(text = "toplimit_center")
    label_toplimit_center.place(x=20, y=181)
    scale_toplimit_center = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500,
                                          command=func_scale_toplimit_center)
    scale_toplimit_center.place(x=130, y=160)
    scale_toplimit_center.set(value=135)

    label_toplimit_width = tkinter.Label(text = "toplimit_width")
    label_toplimit_width.place(x=20, y=251)
    scale_toplimit_width = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500,
                                         command=func_scale_toplimit_width)
    scale_toplimit_width.place(x=130, y=230)
    scale_toplimit_width.set(value=40)

    label_baseline_offset = tkinter.Label(text = "baseline_offset")
    label_baseline_offset.place(x=20, y=321)
    scale_baseline_offset = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500,
                                          command=func_scale_baseline_offset)
    scale_baseline_offset.place(x=130, y=300)
    scale_baseline_offset.set(value=55)

    label_lowerlimt_offset = tkinter.Label(text = "lowerlimt_offset")
    label_lowerlimt_offset.place(x=20, y=391)
    scale_lowerlimt_offset = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500,
                                           command=func_scale_lowerlimt_offset)
    scale_lowerlimt_offset.place(x=130, y=370)
    scale_lowerlimt_offset.set(value=40)

    label_lowerlimt_width = tkinter.Label(text = "lowerlimt_width")
    label_lowerlimt_width.place(x=20, y=461)
    scale_lowerlimt_width = tkinter.Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, 
                                          command=func_scale_lowerlimt_width)
    scale_lowerlimt_width.place(x=130, y=440)
    scale_lowerlimt_width.set(value=35)

    window.mainloop()

def demo_realsense(net, height_size, cpu, pullup_args, realsense_file=None):
    pipeline = rs.pipeline()
    config = rs.config()
    if realsense_file:
        rs.config.enable_device_from_file(config, realsense_file)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline.start(config)

    delay = 1

    preson_heights_ptr = 0
    preson_heights = []

    while True:
        pullup_args.lowerlimit_center = pullup_args.baseline - pullup_args.lowerlimit_offset
        
        frames = pipeline.wait_for_frames()
        frame_color = frames.get_color_frame()
        if not frame_color: continue
        img_bgr = np.asanyarray(frame_color.get_data())
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        orig_img = img.copy()
        current_poses = human_pose_2d(net, img, height_size, cpu)

        if pullup_args.if_draw_line:
            cv2.line(img, (int(0), int(pullup_args.bar_height)), (int(img.shape[1]), int(pullup_args.bar_height)), [255, 255, 255], 2)
            cv2.line(img, (int(0), int(pullup_args.bar_height-pullup_args.bar_width)), (int(img.shape[1]), int(pullup_args.bar_height-pullup_args.bar_width)), [255, 255, 0], 2)
            cv2.line(img, (int(0), int(pullup_args.bar_height+pullup_args.bar_width)), (int(img.shape[1]), int(pullup_args.bar_height+pullup_args.bar_width)), [255, 255, 0], 2)
            cv2.line(img, (int(0), int(pullup_args.toplimit_center-pullup_args.toplimit_width)), (int(img.shape[1]), int(pullup_args.toplimit_center-pullup_args.toplimit_width)), [0, 0, 255], 2)
            cv2.line(img, (int(0), int(pullup_args.toplimit_center+pullup_args.toplimit_width)), (int(img.shape[1]), int(pullup_args.toplimit_center+pullup_args.toplimit_width)), [0, 0, 255], 2)
            cv2.line(img, (int(0), int(pullup_args.baseline)), (int(img.shape[1]), int(pullup_args.baseline)), [255, 0, 0], 2)
            cv2.line(img, (int(0), int(pullup_args.lowerlimit_center-pullup_args.lowerlimit_width)), (int(img.shape[1]), int(pullup_args.lowerlimit_center-pullup_args.lowerlimit_width)), [0, 255, 0], 2)
            cv2.line(img, (int(0), int(pullup_args.lowerlimit_center+pullup_args.lowerlimit_width)), (int(img.shape[1]), int(pullup_args.lowerlimit_center+pullup_args.lowerlimit_width)), [0, 255, 0], 2)
        if pullup_args.if_draw_state:
            text_pullup_cnt = "Current Number = " + str(pullup_args.pullup_cnt)
            cv2.putText(img, text_pullup_cnt, (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            text_pullup_state = "Current State : "
            if pullup_args.pullup_state == 0:
                text_pullup_state = text_pullup_state + "Off-the-Bar"
            elif pullup_args.pullup_state == 1:
                text_pullup_state = text_pullup_state + "Down-to-Up"
            elif pullup_args.pullup_state == 2:
                text_pullup_state = text_pullup_state + "Up-to-Down"
            cv2.putText(img, text_pullup_state, (40, img.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
        
        if not current_poses: continue
        pose = current_poses[0]
        pose.draw(img)
        # off the bar
        if pullup_args.pullup_state == 0:
            if (abs(pullup_args.bar_height - pose.keypoints[4][1]) < pullup_args.bar_width and
                abs(pullup_args.bar_height - pose.keypoints[7][1]) < pullup_args.bar_width):
                pullup_args.baseline = (pose.keypoints[2][1] + pose.keypoints[5][1]) / 2 + pullup_args.baseline_offset
                pullup_args.pullup_state = 1
        # down-to-up
        elif pullup_args.pullup_state == 1:
            if pose.keypoints[2][1] > pullup_args.baseline or pose.keypoints[5][1] > pullup_args.baseline:
                print(f"The final number is {pullup_args.pullup_cnt}.")
                pullup_args.pullup_state = 0
                pullup_args.pullup_cnt = 0 
                preson_heights_ptr = 0
                preson_heights = []
            else:
                if len(preson_heights) < 10:
                    preson_heights.append(pose.keypoints[2][1])
                else:
                    preson_heights[preson_heights_ptr] = pose.keypoints[2][1]
                preson_heights_ptr += 1
                if preson_heights_ptr == 10: preson_heights_ptr = 0
                if len([x for x in preson_heights if abs(pullup_args.toplimit_center - x) < pullup_args.toplimit_width]) > 5:
                    pullup_args.pullup_state = 2
                    pullup_args.pullup_cnt += 1
        # up-to-down
        elif pullup_args.pullup_state == 2:
            if pose.keypoints[2][1] > pullup_args.baseline or pose.keypoints[5][1] > pullup_args.baseline:
                print(f"The final number is {pullup_args.pullup_cnt}.")
                pullup_args.pullup_state = 0
                pullup_args.pullup_cnt = 0 
                preson_heights_ptr = 0
                preson_heights = []
            else:
                if len(preson_heights) < 10:
                    preson_heights.append(pose.keypoints[2][1])
                else:
                    preson_heights[preson_heights_ptr] = pose.keypoints[2][1]
                preson_heights_ptr += 1
                if preson_heights_ptr == 10: preson_heights_ptr = 0
                if len([x for x in preson_heights if abs(pullup_args.lowerlimit_center -x) < pullup_args.lowerlimit_width]) > 3:
                    pullup_args.pullup_state = 1
                

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))        
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        # fps = 1/(time.time()-last_time)
        # print(f'fps = {fps}')
        # print(img.shape)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='''Lightweight human pose estimation python demo.
                       This is just for quick results preview.
                       Please, consider c++ demo for the best performance.''')
    parser.add_argument('--checkpoint-path', type=str, required=True, help='path to the checkpoint')
    parser.add_argument('--height-size', type=int, default=256, help='network input layer height size')
    parser.add_argument('--video', type=str, default='', help='path to video file or camera id')
    parser.add_argument('--images', nargs='+', default='', help='path to input image(s)')
    parser.add_argument('--cpu', action='store_true', help='run network inference on cpu')
    parser.add_argument('--realsense', type=str, default='', help="Path to the bag file or 0(Camera)")
    args = parser.parse_args()

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)
    net = net.eval()
    if not args.cpu:
        net = net.cuda()
    
    if args.images:
        demo_images(net, args.images, args.height_size, args.cpu)
    elif args.video:
        demo_video(net, args.video, args.height_size, args.cpu)
    elif args.realsense:
        pullup_args = PullupArgs(bar_height=130,
                             bar_width=40,
                             toplimit_center=135,
                             toplimit_width=40,
                             lowerlimit_offset=40,
                             lowerlimit_width=35,
                             baseline_offset=55,
                             if_draw_line=1,
                             if_draw_state=1,
                            )
        realsense_path = None if args.realsense == '0' else args.realsense
        thread_gui_pullup_args = threading.Thread(target=gui_pullup_args, args=(pullup_args,))
        thread_demo_realsense = threading.Thread(target=demo_realsense, args=(net, args.height_size, args.cpu, pullup_args, realsense_path))
        thread_gui_pullup_args.start()
        thread_demo_realsense.start()
        # demo_realsense(net, args.height_size, args.cpu, pullup_args, args.realsense_file)
    else:
        raise ValueError('No source has to be provided!')

    # python demo_decouple.py --checkpoint-path checkpoint/checkpoint_iter_370000.pth --realsense data/12-1-1280.bag