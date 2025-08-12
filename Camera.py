import cv2
from PyQt5.QtCore import QThread
import Bridge
import toml, os, datetime
import numpy as np

class Camera(QThread):
    def __init__(self, debuger=None):
        super().__init__()

        # 初始化参数
        self.config = toml.load("./config/config.toml")
        self.camera_config = toml.load(os.path.join("./config", self.config["specific_config"]))["camera"]
        self.K = np.array(self.camera_config["K"], dtype=np.float32)
        self.D = np.array(self.camera_config["D"], dtype=np.float32)
        self.alpha = self.camera_config["alpha"]
        self.image_shape = self.camera_config["image_shape"]
        self.rotate = self.camera_config["rotate"]
        self.show_icon = self.camera_config["show_icon"]
        ROOT = os.path.dirname(os.path.abspath(__file__))
        self.record_path = os.path.join(ROOT, self.camera_config["record_path"], str(datetime.datetime.now()).replace(':', '-').replace(' ', '_') + ".mp4")
        self.record_path = self.record_path.replace('/', os.sep).replace("\\", os.sep)
        self.record = self.camera_config["record"]

        self.stop = False
        self.need_restart = False
        self.debuger = debuger

        self.new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, self.image_shape, self.alpha, self.image_shape, True)

        self.mapX, self.mapY = cv2.fisheye.initUndistortRectifyMap(self.K, self.D, np.eye(3), self.new_K, self.image_shape, cv2.CV_16SC2) # 去畸变
        print(f"{self.K = }\n{self.new_K = }")

        # 初始化线程间通信
        self.publisher = Bridge.Publisher("image")
        self.publisher_log = Bridge.Publisher("log")
        if self.show_icon:
            self.publisher_icon = Bridge.Publisher("icon")

        # 初始化视屏流 确保打开视频
        while True:
            # 这个参数指定很重要，之前打开相机驱动所需时间太久就是因为默认参数打开的相机驱动不合适
            # CAP_ANY：只能读一张图，要启动很久；CAP_DSHOW：很快，有概率雪花屏；CAP_MSMF：同CAP_ANY
            # 第一处不同
            self.video = cv2.VideoCapture(self.camera_config["id"], cv2.CAP_DSHOW)
            if self.video.isOpened():
                break
            else:
                print("Fail to open video!")
        #  设置视频的大小
        # 指定（就是视频的一帧一帧的图片全部都变成jpg格式

        self.video.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if self.record:
            self.writer = cv2.VideoWriter(self.record_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, self.image_shape, True)

    def run(self):
        while not self.stop:
            if self.need_restart:
                self.video.release()
                while True:
                    self.video = cv2.VideoCapture(self.camera_config["id"], cv2.CAP_DSHOW)
                    self.video.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
                    self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    if self.video.isOpened():
                        self.need_restart = False
                        self.publisher_log.publish("成功重启相机")
                        break
                    else:
                        print("Fail to open video!")
                        self.publisher_log.publish("重启相机失败")

            # 获取一帧的图片
            try:
                ret, now_image = self.video.read() # ret来判断是否读取成功
            except:
                print("Fail to read image from the capture!")
                continue
            
            if not ret:
                continue 
            
            if self.show_icon:
                # 裁剪出徽标，显示图传接收端是否在正常工作
                icon = np.array(now_image[0:25, 0:25, ::-1])
                icon = cv2.resize(icon, (100, 100))
                self.publisher_icon.publish(icon)

            if self.rotate:
                now_image = cv2.rotate(now_image, cv2.ROTATE_180)
            # 裁剪拉伸规范化图片 #先高后宽，虽然无所谓（可是会裁剪掉宽或者高的一部分（具体来说是和高不一致的部分
            ratio_x = now_image.shape[1] / self.image_shape[0]
            ratio_y = now_image.shape[0] / self.image_shape[1]
            if ratio_x > ratio_y:
                now_image = now_image[:, int((now_image.shape[1] - ratio_y * self.image_shape[0]) / 2):
                                    int((now_image.shape[1] + ratio_y * self.image_shape[0]) / 2), :].copy()
            else:
                now_image = now_image[int((now_image.shape[0] - ratio_x * self.image_shape[1]) / 2):
                                    int((now_image.shape[0] + ratio_x * self.image_shape[1]) / 2), :, :].copy()
            now_image = cv2.resize(now_image, self.image_shape)

            # 矫正畸变
            img_undistorted = cv2.remap(now_image, self.mapX, self.mapY, cv2.INTER_LINEAR)
            # cv2.imshow("camera", img_undistorted)
            # cv2.waitKey(1)
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
        
    