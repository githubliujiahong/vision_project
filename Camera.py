"""
@Module: Camera
@Description: 视觉系统的图像采集与预处理模块。
              负责从本地视频流或物理相机获取图像，并进行 Letterbox 无损缩放（Padding），
              将画面规范化为 YOLO 模型所需的标准输入尺寸 (默认 640x640)，最后发布给后续处理模块。
"""

import cv2
from PyQt5.QtCore import QThread
import Bridge
import toml, os
import numpy as np
import time


class Camera(QThread):
    def __init__(self, debuger=None) -> None:
        """
        初始化相机模块，加载配置项并建立通信发布者。
        """
        super().__init__()

        # --- 1. 加载级联配置文件 ---
        self.config = toml.load("./config/config.toml")
        self.camera_config = toml.load(os.path.join("./config", self.config["specific_config"]))["camera"]

        # --- 2. 初始化核心参数 ---
        # 目标输入尺寸，通常为 [640, 640]
        self.image_shape = self.camera_config.get("image_shape", [640, 640])
        self.rotate = self.camera_config.get("rotate", False)
        self.show_icon = self.camera_config.get("show_icon", False)

        # --- 3. 初始化录制功能 ---
        self.record = self.camera_config.get("record", False)
        if self.record:
            # 使用当前时间戳生成唯一的视频文件名，防止覆盖
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            file_name = f'./record/output_{int(time.time())}.mp4'
            self.writer = cv2.VideoWriter(file_name, fourcc, 30.0, tuple(self.image_shape))

        # --- 4. 线程控制与通信总线 ---
        self.stop = False
        self.need_restart = False
        self.debuger = debuger
        self.video = None

        # 注册向外发送数据的 Publisher
        self.publisher = Bridge.Publisher("image")
        self.publisher_log = Bridge.Publisher("log")
        if self.show_icon:
            self.publisher_icon = Bridge.Publisher("icon")

        # 启动相机连接
        self._connect_camera()

    def _connect_camera(self) -> None:
        """
        根据配置文件中的 record_path 智能连接数据源 (物理相机或本地视频)。
        """
        video_path = self.camera_config.get("record_path", "./record/record.mp4")

        # 情况 A: 如果是纯数字(如 0 或 "0")，作为物理 USB 摄像头打开
        if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
            self.video = cv2.VideoCapture(int(video_path))
            source_type = "Camera Device"

        # 情况 B: 如果是文件路径且存在，作为视频文件打开（模拟测试）
        elif isinstance(video_path, str) and os.path.exists(video_path):
            self.video = cv2.VideoCapture(video_path)
            source_type = "Local Video"

        # 情况 C: 配置错误，强行使用默认摄像头 0 兜底，并发送警告日志
        else:
            err_msg = f"[Camera Warn] 数据源 {video_path} 无效，强制使用默认相机 0"
            print(err_msg)
            self.publisher_log.publish(err_msg)  # 让 UI 界面也能看到报错
            self.video = cv2.VideoCapture(0)
            source_type = "Fallback Camera 0"

        # 最终校验是否成功打开
        if not self.video.isOpened():
            fatal_msg = "[Camera Error] 致命错误：无法打开任何视频流！"
            print(fatal_msg)
            self.publisher_log.publish(fatal_msg)
        else:
            print(f"[Camera] 成功连接至: {source_type}")

    def run(self) -> None:
        """
        相机线程的主循环。负责拉取帧、Letterbox 预处理以及发布。
        """
        target_w, target_h = self.image_shape[0], self.image_shape[1]

        while not self.stop:
            # 1. 处理外部触发的重启请求
            if self.need_restart:
                if self.video:
                    self.video.release()
                self._connect_camera()
                self.need_restart = False

            # 2. 安全地读取一帧图像
            try:
                if self.video is None:
                    time.sleep(0.1)
                    continue
                ret, now_image = self.video.read()
            except Exception as e:
                print(f"[Camera Error] 读取图像帧失败: {e}")
                time.sleep(0.05)
                continue

            # 3. 处理视频播放结束的情况 (循环播放机制)
            if not ret:
                self.publisher_log.publish("本地视频播放完毕，正在重置循环...")
                self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if now_image is None or now_image.size == 0:
                continue

            # 4. 发布左上角的 UI 小图标 (可选功能)
            if self.show_icon:
                h, w = now_image.shape[:2]
                if h > 25 and w > 25:
                    icon = np.array(now_image[0:25, 0:25, ::-1])
                    icon = cv2.resize(icon, (100, 100))
                    self.publisher_icon.publish(icon)

            # 5. 图像倒置 (如果相机倒装)
            if self.rotate:
                now_image = cv2.rotate(now_image, cv2.ROTATE_180)

            # ==========================================================
            # 6. 核心预处理：Letterbox (等比例无损缩放与灰边填充)
            # ==========================================================
            orig_h, orig_w = now_image.shape[:2]

            # 6.1 计算最小缩放比例，确保长边刚好贴合目标尺寸，短边留白
            scale = min(target_w / orig_w, target_h / orig_h)

            # 6.2 计算缩放后的实际新尺寸 (必定有一条边等于 640)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            if new_w <= 0 or new_h <= 0:
                continue

            # 6.3 执行等比例缩放
            img_resized = cv2.resize(now_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 6.4 计算需要填充的灰边厚度 (处理奇数情况：一侧多一像素)
            pad_w = target_w - new_w
            pad_h = target_h - new_h

            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            # 6.5 使用 114 灰色值进行边缘填充 (YOLO 标准背景色)
            img_letterbox = cv2.copyMakeBorder(
                img_resized,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114)
            )
            # ==========================================================

            # 7. 发布处理好的标准图像，供 YOLO 和 UI 使用
            self.publisher.publish(img_letterbox, timeout=0.01)

            # 8. 录制保存
            if self.record and hasattr(self, 'writer'):
                self.writer.write(img_letterbox)

            # 略微休眠，让出 CPU 切片
            time.sleep(0.03)

        # 循环结束后的资源释放清理
        if self.video:
            self.video.release()
        if self.record and hasattr(self, 'writer'):
            self.writer.release()

    def restart(self) -> None:
        """从外部触发相机重启的标志位"""
        self.need_restart = True

    def end(self) -> None:
        """从外部安全终止相机线程"""
        self.stop = True