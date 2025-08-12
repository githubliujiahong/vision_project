import Bridge
import toml
import torch
import time
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGroupBox, QDialog,QApplication
from PyQt5.QtCore import QThread


class Debuger(QThread):
    def __init__(self):
        super().__init__()
        print("Start debuger init.")

        self.config = toml.load("./config/debug.toml") 
        self.detect_res = torch.tensor([])
        self.last_message_time = {"image":time.time(), "detect_res":time.time()}
        from post_process.Post_process import Mean_filter
        self.message_fps = {"image":Mean_filter(5), "detect_res":Mean_filter(5)} # 平均滤波处理
        self.stop = False
        self.window = Debug_window()
        self.param_dic = {}
        
        self.window.show()

        self.image_sub = Bridge.Subscriber("image")
        self.detect_sub = Bridge.Subscriber("detect_res")
        print("Debuger init finished.")

    def run(self):
        while not self.stop:
            # 接收各话题消息
            try:
                self.image_sub.get_message(0.01)
                # now_time=现在的时间
                now_time = time.time()
                self.message_fps["image"].push(1 / (now_time - self.last_message_time["image"])) # 更新滤波器，判断识别框更新的快慢，加入一个新的数值
                self.last_message_time["image"] = now_time
            except:
                pass
            try:
                self.detect_sub.get_message(0.01)
                now_time = time.time()
                self.message_fps["detect_res"].push(1 / (now_time - self.last_message_time["detect_res"]))
                self.last_message_time["detect_res"] = now_time
            except:
                pass
            
            message_list = self.message_fps.keys()
            fps_text = ""
            for key in message_list:
                fps_text += f"{key}: {self.message_fps[key].get_average()}"
            self.window.update_fps(fps_text)

            param_list = self.param_dic.keys()
            param_text = ""
            for key in param_list:
                param_text += f"{key}: {round(self.param_dic[key].get_average(), 2)}\n"
            self.window.update_param(param_text)
            

    # 我们可以认为，这个东西是得到一个param里的滤波器
    def add_param(self, name, queue_length=10):
        from post_process.Post_process import Mean_filter
        self.param_dic[name] = Mean_filter(queue_length)
        
    def update_param(self, name, data):
        self.param_dic[name].push(data)

    def end(self):
        self.stop = True

class Debug_window(QDialog):
    def __init__(self):
        super().__init__()

        # 子UI初始化
        self.setWindowTitle("Dbug")

        param_group_box = QGroupBox("参数打印")
        self.param_label = QLabel("")
        param_layout = QVBoxLayout()
        param_layout.addWidget(self.param_label)
        param_group_box.setLayout(param_layout)

        fps_group_box = QGroupBox("话题帧率测试")
        self.fps_label = QLabel("")
        fps_layout = QVBoxLayout()
        fps_layout.addWidget(self.fps_label)
        fps_group_box.setLayout(fps_layout)

        layout = QVBoxLayout()
        layout.addWidget(fps_group_box)
        layout.addWidget(param_group_box)
        self.setLayout(layout)
    
    def update_param(self, text):
        self.param_label.setText(text)

    def update_fps(self, text):
        pass
        # print(text)
        # self.fps_label.setText(text)

