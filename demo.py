import argparse
from glob import glob

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import time
import os.path
import copy
import threading

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
from val import normalize, pad_width

from tkinter import *


class ImageReader(object):
    def __init__(self, file_names):
        self.file_names = file_names
        self.max_idx = len(file_names)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == self.max_idx:
            raise StopIteration
        img = cv2.imread(self.file_names[self.idx], cv2.IMREAD_COLOR)
        if img.size == 0:
            raise IOError('Image {} cannot be read'.format(self.file_names[self.idx]))
        self.idx = self.idx + 1
        return img


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


def run_demo_rs(net, height_size, cpu, track, smooth, inputdir=None):
    pipeline = rs.pipeline()
    config = rs.config()
    if inputdir:
        rs.config.enable_device_from_file(config, inputdir)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
    pipeline.start(config)
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1

    global Bar_Height      # !!!
    global Bar_Height_rg   # !!!
    global up_Height       # !!!
    global up_Height_rg    # !!!
    global checkHeight_b   # !!!
    global down_Height_b   # !!!
    global down_Height_rg  # !!!
    global ifDraw
    checkHeight = 0
    down_Height = 0

    Person_State = 0 # off the bar
    Bar_cnt = 0
    
    ptr_check = 0
    wind_check = []

    start_time = 0
    timemap = []
    tmp_time = [0, 0]

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img_bgr = np.asanyarray(color_frame.get_data())
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        last_time = time.time()

        orig_img = img.copy()
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

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses

            
        for pose in current_poses:
            down_Height = checkHeight - down_Height_b
            if ifDraw:
                cv2.line(img, (int(0), int(Bar_Height)), (int(img.shape[1]), int(Bar_Height)), [255, 255, 255], 2)
                cv2.line(img, (int(0), int(Bar_Height-Bar_Height_rg)), (int(img.shape[1]), int(Bar_Height-Bar_Height_rg)), [255, 255, 0], 2)
                cv2.line(img, (int(0), int(Bar_Height+Bar_Height_rg)), (int(img.shape[1]), int(Bar_Height+Bar_Height_rg)), [255, 255, 0], 2)
                cv2.line(img, (int(0), int(up_Height-up_Height_rg)), (int(img.shape[1]), int(up_Height-up_Height_rg)), [0, 0, 255], 2)
                cv2.line(img, (int(0), int(up_Height+up_Height_rg)), (int(img.shape[1]), int(up_Height+up_Height_rg)), [0, 0, 255], 2)
                cv2.line(img, (int(0), int(checkHeight)), (int(img.shape[1]), int(checkHeight)), [255, 0, 0], 2)
                cv2.line(img, (int(0), int(down_Height-down_Height_rg)), (int(img.shape[1]), int(down_Height-down_Height_rg)), [0, 255, 0], 2)
                cv2.line(img, (int(0), int(down_Height+down_Height_rg)), (int(img.shape[1]), int(down_Height+down_Height_rg)), [0, 255, 0], 2)
            if ifDrawState:
                text_cnt = "Current Number = " + str(Bar_cnt)
                cv2.putText(img, text_cnt, (40, 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
                text_state = "Current State : "
                if Person_State == 0:
                    text_state = text_state + "Off-the-Bar"
                elif Person_State == 1:
                    text_state = text_state + "Down-to-Up"
                elif Person_State == 2:
                    text_state = text_state + "Up-to-Down"
                cv2.putText(img, text_state, (40, img.shape[0] - 50), cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 0, 255), 2)
            pose.draw(img)

            if Person_State == 0: # off the bar
                if pose.res(img, Bar_Height, pose.pheight(4)[1], Bar_Height_rg) and pose.res(img, Bar_Height, pose.pheight(7)[1], Bar_Height_rg):
                    Person_State = 1
                    checkHeight = (pose.pheight(2)[1] + pose.pheight(5)[1]) / 2 + checkHeight_b
                    start_time = time.time()
                    timemap = []
                    tmp_time = [0, 0]
                    tmp_time[0] = time.time() - start_time
                    
                    
            elif Person_State == 1: # down-to-up
                if pose.pheight(2)[1] > checkHeight or pose.pheight(5)[1] > checkHeight:
                    print(f"The final number is {Bar_cnt}.")
                    Person_State = 0 
                    ptr_check = 0
                    wind_check = []
                    Bar_cnt = 0
                else:
                    if len(wind_check) < 10:
                        wind_check.append(pose.pheight(2)[1])
                        ptr_check += 1
                        if ptr_check == 10:
                            ptr_check = 0
                    else:
                        wind_check[ptr_check] = pose.pheight(2)[1]
                        ptr_check += 1
                        if ptr_check == 10:
                            ptr_check = 0

                    cnt_list = [x for x in wind_check if pose.res(img, up_Height, x, up_Height_rg)]
                    if len(cnt_list) > 5:
                        Person_State = 2
                        Bar_cnt += 1
                        tmp_time[1] = time.time() - start_time
                        timemap.append(copy.deepcopy(tmp_time))

            elif Person_State == 2: # up-to-down
                if pose.pheight(2)[1] > checkHeight or pose.pheight(5)[1] > checkHeight:
                    print(f"The final number is {Bar_cnt}.")
                    Person_State = 0
                    ptr_check = 0
                    wind_check = []
                    Bar_cnt = 0
                else:
                    if len(wind_check) < 10:
                        wind_check.append(pose.pheight(2)[1])
                        ptr_check += 1
                        if ptr_check == 10:
                            ptr_check = 0
                    else:
                        wind_check[ptr_check] = pose.pheight(2)[1]
                        ptr_check += 1
                        if ptr_check == 10:
                            ptr_check = 0

                    cnt_list = [x for x in wind_check if pose.res(img, down_Height, x, down_Height_rg)]
                    if len(cnt_list) > 3:
                        Person_State = 1
                        tmp_time[0] = time.time() - start_time
                

        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        
        # fps = 1/(time.time()-last_time)
        # print(f'fps = {fps}')
        # print(img.shape)
        
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


def run_demo(net, image_provider, height_size, cpu, track, smooth):
    net = net.eval()
    if not cpu:
        net = net.cuda()

    stride = 8
    upsample_ratio = 4
    num_keypoints = Pose.num_kpts
    previous_poses = []
    delay = 1
    for img in image_provider:
        orig_img = img.copy()
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

        if track:
            track_poses(previous_poses, current_poses, smooth=smooth)
            previous_poses = current_poses
        for pose in current_poses:
            pose.draw(img)
        img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
        for pose in current_poses:
            cv2.rectangle(img, (pose.bbox[0], pose.bbox[1]),
                          (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]), (0, 255, 0))
            if track:
                cv2.putText(img, 'id: {}'.format(pose.id), (pose.bbox[0], pose.bbox[1] - 16),
                            cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('Lightweight Human Pose Estimation Python Demo', img)
        key = cv2.waitKey(delay)
        if key == 27:  # esc
            return
        elif key == 112:  # 'p'
            if delay == 1:
                delay = 0
            else:
                delay = 1


def main():
    global args
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    load_state(net, checkpoint)

    if args.inputbag != '':
        args.track = 1
        run_demo_rs(net, args.height_size, args.cpu, args.track, args.smooth, args.inputbag)
    elif args.realsense:
        args.track = 1
        run_demo_rs(net, args.height_size, args.cpu, args.track, args.smooth)
    else:
        frame_provider = ImageReader(args.images)
        if args.video != '':
            frame_provider = VideoReader(args.video)
        else:
            args.track = 0

        run_demo(net, frame_provider, args.height_size, args.cpu, args.track, args.smooth)


def func_scale_Bar_Height(v):
    global Bar_Height
    Bar_Height = int(v)
def func_scale_Bar_Height_rg(v):
    global Bar_Height_rg
    Bar_Height_rg = int(v)
def func_scale_up_Height(v):
    global up_Height
    up_Height = int(v)
def func_scale_up_Height_rg(v):
    global up_Height_rg
    up_Height_rg = int(v)
def func_scale_checkHeight_b(v):
    global checkHeight_b
    checkHeight_b = int(v)
def func_scale_down_Height_b(v):
    global down_Height_b
    down_Height_b = int(v)
def func_scale_down_Height_rg(v):
    global down_Height_rg
    down_Height_rg = int(v)
def func_button_ifDraw():
    global ifDraw
    global button_ifDraw
    if ifDraw:
        ifDraw = 0
        button_ifDraw['text'] = "显示测试曲线"
    else:
        ifDraw = 1
        button_ifDraw['text'] = "不显示测试曲线"
def func_button_ifDrawState():
    global ifDrawState
    global button_ifDrawState
    if ifDrawState:
        ifDrawState = 0
        button_ifDrawState['text'] = "显示测试信息"
    else:
        ifDrawState = 1
        button_ifDrawState['text'] = "不显示测试信息"
           

def console_GUI():
    window = Tk()
    window.title("BDMI2022s")
    window.geometry("1280x580")

    label_explanation0 = Label(text = "参数说明：")
    label_explanation0.place(x=650, y=41)
    label_explanation1 = Label(text = "单杠高度为Bar_Height，上杠条件为手部高度在[Bar_Height-Bar_Height_rg, Bar_Height+Bar_Height_rg]范围内")
    label_explanation1.place(x=650, y=111)
    label_explanation2 = Label(text = "肩部的初始高度为checkHeight+checkHeight_b，下杠判断为肩膀高度 > checkHeight")
    label_explanation2.place(x=650, y=181)
    label_explanation3 = Label(text = "完成一个上升的标志为肩膀高度进入[up_Height-up_Height_rg, up_Height_up_Height_rg]区间")
    label_explanation3.place(x=650, y=251)
    label_explanation4 = Label(text = "完成一个下降的标志为肩膀高度进入[down_Height-down_Height_rg, down_Height+down_Height_rg]区间")
    label_explanation4.place(x=650, y=321)
    label_explanation5 = Label(text = "其中down_Height = checkHeight - down_Height_b")
    label_explanation5.place(x=650, y=391)
    global button_ifDraw
    button_ifDraw = Button(window, text="不显示测试曲线", command=func_button_ifDraw)
    button_ifDraw.place(x=650, y=458)
    global button_ifDrawState
    button_ifDrawState = Button(window, text="不显示测试信息", command=func_button_ifDrawState)
    button_ifDrawState.place(x=850, y=458)


    label_Bar_Height = Label(text = "Bar_Height")
    label_Bar_Height.place(x=20, y=41)
    
    scale_Bar_Height = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_Bar_Height)
    scale_Bar_Height.place(x=130, y=20)
    scale_Bar_Height.set(value=130)

    label_Bar_Height_rg = Label(text = "Bar_Height_rg")
    label_Bar_Height_rg.place(x=20, y=111)
    
    scale_Bar_Height_rg = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_Bar_Height_rg)
    scale_Bar_Height_rg.place(x=130, y=90)
    scale_Bar_Height_rg.set(value=40)

    label_up_Height = Label(text = "up_Height")
    label_up_Height.place(x=20, y=181)
    
    scale_up_Height = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_up_Height)
    scale_up_Height.place(x=130, y=160)
    scale_up_Height.set(value=135)

    label_up_Height_rg = Label(text = "up_Height_rg")
    label_up_Height_rg.place(x=20, y=251)
    
    scale_up_Height_rg = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_up_Height_rg)
    scale_up_Height_rg.place(x=130, y=230)
    scale_up_Height_rg.set(value=40)

    label_checkHeight_b = Label(text = "checkHeight_b")
    label_checkHeight_b.place(x=20, y=321)
    
    scale_checkHeight_b = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_checkHeight_b)
    scale_checkHeight_b.place(x=130, y=300)
    scale_checkHeight_b.set(value=55)

    label_down_Height_b = Label(text = "down_Height_b")
    label_down_Height_b.place(x=20, y=391)
    
    scale_down_Height_b = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_down_Height_b)
    scale_down_Height_b.place(x=130, y=370)
    scale_down_Height_b.set(value=40)

    label_down_Height_rg = Label(text = "down_Height_rg")
    label_down_Height_rg.place(x=20, y=461)
    
    scale_down_Height_rg = Scale(window, from_=0, to=960, orient='horizonta', tickinterval=100, length=500, \
                            command=func_scale_down_Height_rg)
    scale_down_Height_rg.place(x=130, y=440)
    scale_down_Height_rg.set(value=35)

    window.mainloop()

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
    parser.add_argument('--track', type=int, default=1, help='track pose id in video')
    parser.add_argument('--smooth', type=int, default=1, help='smooth pose keypoints')
    parser.add_argument('--realsense',action='store_true')
    parser.add_argument('--inputbag', type=str, default='', help="Path to the bag file")
    args = parser.parse_args()

    # print('debug!')

    if args.video == '' and args.images == '' and not args.realsense and args.inputbag == '':
        raise ValueError('Either --video or --image or --realsense has to be provided')

    ## 有!!!标记的变量为需要预设的超参数
    ## 单杠高度为Bar_Height，上杠条件为手部高度在[Bar_Height-Bar_Height_rg, Bar_Height+Bar_Height_rg]范围内
    ## 肩部的初始高度为checkHeight+checkHeight_b，下杠判断为肩膀高度 > checkHeight
    ## 完成一个上升的标志为肩膀高度进入[up_Height-up_Height_rg, up_Height_up_Height_rg]区间
    ## 完成一个下降的标志为肩膀高度进入[down_Height-down_Height_rg, down_Height+down_Height_rg]区间
    ## 其中down_Height = checkHeight - down_Height_b

    Bar_Height = 130    # !!!
    Bar_Height_rg = 40  # !!!
    up_Height = 135     # !!!
    up_Height_rg = 40   # !!!
    checkHeight_b = 55  # !!!
    down_Height_b = 40  # !!!
    down_Height_rg = 35 # !!!
    ifDraw = 1
    ifDrawState = 1

    thread_main = threading.Thread(target=main)
    thread_console = threading.Thread(target=console_GUI)
    thread_console.start()
    thread_main.start()
    
