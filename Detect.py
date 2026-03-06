"""
@Module: Detector
@Description: 视觉系统的目标检测大脑。
              采用双模型架构（主模型识别魔方/盒子，球模型专门识别台球）。
              利用 PyTorch 和 CUDA 硬件加速进行推理，并将两个模型的张量结果
              合并映射到全局统一的 ID 体系中，供下游决策模块使用。
"""

from PyQt5.QtCore import QThread
import Bridge
import toml
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time


class Detector(QThread):
    def __init__(self, debuger=None) -> None:
        super().__init__()

        # --- 1. 配置加载与总线注册 ---
        self.config = toml.load("./config/detect.toml")
        self.input_shape = self.config["input_shape"]

        self.subscriber = Bridge.Subscriber("image")
        self.publisher = Bridge.Publisher("detect_res")
        self.publisher_log = Bridge.Publisher("log")  # 新增日志发布者

        self.stop = False
        self.debuger = debuger

        # --- 2. 硬件加速检测与模型加载 ---
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        init_msg = f"[Detector] 正在初始化 YOLO 推理引擎，计算平台: {self.device.upper()}"
        print(init_msg)
        self.publisher_log.publish(init_msg)

        # 加载主模型（负责检测魔方、盒子等环境物体）
        self.model = YOLO(self.config["path"])
        self.model.to(self.device)

        # 加载专属球模型（专门负责高精度台球识别）
        self.model_ball = YOLO(self.config["path_ball"])
        self.model_ball.to(self.device)

        # --- 3. 模型显存预热 (Warm-up) ---
        # 深度学习模型在第一次推理时通常较慢，塞入一张纯黑空图进行预热，防止运行时掉帧
        self.publisher_log.publish("[Detector] 正在进行张量网络预热...")
        dummy_img = np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8)
        _ = self.model.predict(dummy_img, verbose=False)
        _ = self.model_ball.predict(dummy_img, verbose=False)

        self.publisher_log.publish("[Detector] 初始化完成，已进入备战状态！")

    def run(self) -> None:
        """
        检测线程主循环：拉取图像 -> 双模型并行推理 -> 结果融合 -> 发布
        """
        while not self.stop:
            try:
                # 1. 获取最新处理好的图像 (非阻塞式读取)
                now_image = self.subscriber.get_message(0.005)
                if now_image is None:
                    continue

                # 2. 颜色空间转换 (YOLO 训练时使用的是 RGB，而 OpenCV 默认读图是 BGR)
                im = cv2.cvtColor(now_image, cv2.COLOR_BGR2RGB)

                # 3. 核心推理过程 (Dual-Model Inference)
                # max_det=1000: 防止画面过于混乱时漏检
                results = self.model.predict(
                    im, conf=0.25, iou=0.45, max_det=1000, imgsz=self.input_shape, verbose=False
                )
                results_ball = self.model_ball.predict(
                    im, conf=0.25, iou=0.45, max_det=1000, imgsz=self.input_shape, verbose=False
                )

                # 4. 提取底层 Bounding Box 数据
                boxes = results[0].boxes if len(results) > 0 else None
                boxes_ball = results_ball[0].boxes if len(results_ball) > 0 else None

                # 5. 合并并校准数据
                numpy_like = self._merge_detection_results(boxes, boxes_ball)

                # 6. 将标准的 [N, 6] 矩阵发布到消息总线，供 Post_process 决策
                self.publisher.publish(numpy_like, 0.01)

                time.sleep(0.01)

            except Exception as e:
                err_msg = f"[Detector Error] 推理过程崩溃: {e}"
                print(err_msg)
                self.publisher_log.publish(err_msg)
                time.sleep(0.05)  # 出错后主动降频，防止日志刷屏

    def _merge_detection_results(self, boxes, boxes_ball) -> np.ndarray:
        """
        融合双模型的预测结果，并进行全局 ID 对齐映射。

        全局定义 (README): 0:cube, 1:yellow, 2:green, 3:brown, 4:blue, 5:pink, 6:black
        主模型预测: 0:cube
        球模型预测: 0:yellow, 1:green ... -> 需要将 ID 统一加上偏移量 1
        """
        list_to_merge = []

        # --- 处理主模型 (魔方/盒子) ---
        if boxes is not None and len(boxes) > 0:
            data_main = boxes.data.cpu().numpy()
            list_to_merge.append(data_main)

        # --- 处理球模型 (台球) ---
        if boxes_ball is not None and len(boxes_ball) > 0:
            # 必须使用 .copy()，避免直接修改 PyTorch 底层张量数据引发内存污染
            data_ball = boxes_ball.data.cpu().numpy().copy()

            # ID 映射校准：将球模型的结果偏移至全局对应 ID
            # 如果球模型认出的 0 是黄球，加上偏移量 1，就变成了全局的 1 (黄球)
            id_offset = 1
            data_ball[:, 5] += id_offset

            list_to_merge.append(data_ball)

        # --- 矩阵拼接 ---
        if len(list_to_merge) > 0:
            # vstack 将多个 [N, 6] 矩阵垂直堆叠为一个大矩阵
            return np.vstack(list_to_merge)
        else:
            # 如果画面绝对干净，返回空的安全矩阵防止报错
            return np.empty((0, 6), dtype=np.float32)

    def end(self) -> None:
        """安全终止检测线程"""
        self.stop = True