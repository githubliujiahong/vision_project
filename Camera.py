import cv2
from PyQt5.QtCore import QThread
import Bridge
import toml, os, datetime
import numpy as np
import time

class Camera(QThread):
    def __init__(self, debuger=None):
        super().__init__()

        self.config = toml.load("./config/config.toml")
        self.camera_config = toml.load(os.path.join("./config", self.config["specific_config"]))["camera"]
        self.K = np.array(self.camera_config["K"], dtype=np.float32)
        self.D = np.array(self.camera_config["D"], dtype=np.float32)
        self.alpha = self.camera_config["alpha"]
        self.image_shape = self.camera_config["image_shape"]
        self.rotate = self.camera_config["rotate"]
        self.show_icon = self.camera_config["show_icon"]

        self.stop = False
        self.need_restart = False
        self.debuger = debuger

        self.new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, self.image_shape, self.alpha, self.image_shape, True)

        self.mapX, self.mapY = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.new_K, self.image_shape, cv2.CV_16SC2) 
        print(f"{self.K = }\n{self.new_K = }")


        self.publisher = Bridge.Publisher("image")
        self.publisher_log = Bridge.Publisher("log")
        if self.show_icon:
            self.publisher_icon = Bridge.Publisher("icon")


    def _connect_camera(self):
            """
            独立出来的相机连接函数(用于重连)
            """
            

    def run(self):
        while not self.stop:
            if self.need_restart:
                self.video.release()
                self._connect_camera() 
                self.need_restart = False

            # 获取一帧的图片
            try:
                ret, now_image = self.video.read() 
            except:
                print("Fail to read image from the capture!")
                continue
            
            if not ret:
                continue 
            
            if self.show_icon:
                icon = np.array(now_image[0:25, 0:25, ::-1])
                icon = cv2.resize(icon, (100, 100))
                self.publisher_icon.publish(icon)

            if self.rotate:
                now_image = cv2.rotate(now_image, cv2.ROTATE_180)
            # 裁剪拉伸规范化图片 
            """第二个填空点，填写图像的裁剪拉伸规范化 """
            

            img_undistorted = cv2.remap(now_image, self.mapX, self.mapY, cv2.INTER_LINEAR)

            self.publisher.publish(img_undistorted, 0.01)
            if self.record:
                self.writer.write(img_undistorted)
        self.video.release()
        if self.record:
            self.writer.release()

    def restart(self):
        self.need_restart = True
    
    def end(self):
        self.stop = True
        
    