import tkinter

def gui_pullup_args(pullup_args: PullupArgs):
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
