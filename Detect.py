from PyQt5.QtCore import QThread
import Bridge
import toml
import cv2
import numpy as np
from ultralytics import YOLO

class Detector(QThread):
    def __init__(self, debuger=None) -> None:
        print("start to init detector!")
        super().__init__()

        self.config = toml.load("./config/detect.toml")
        self.input_shape = self.config["input_shape"]
        self.subscriber = Bridge.Subscriber("image")
        self.publisher = Bridge.Publisher("detect_res")
        self.stop = False
        self.debuger = debuger
        
        self.model = YOLO(self.config["path"])
        self.model_ball = YOLO(self.config["path_ball"])
        print(f"Using main model: {self.config['path']}")
        print(f"Using ball model: {self.config['path_ball']}")

        self.use_cuda = self.model.device.type == 'cuda'
        print(f"use cuda: {self.use_cuda}")
        
        # 预热
        _ = self.model.predict(np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8), verbose=False)
        _ = self.model_ball.predict(np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8), verbose=False)
        print("detector init finished!")

    def run(self):
        while not self.stop:
            try:
                now_image = self.subscriber.get_message(0.001)
            except:
                # print("Detector receive no image!")
                continue
                
            im = cv2.cvtColor(now_image, cv2.COLOR_BGR2RGB)
            
            # 使用两个模型进行检测
            results = self.model.predict(
                im, 
                conf=0.25, 
                iou=0.45, 
                max_det=1000,
                imgsz=self.input_shape,
                verbose=False
            )
            
            results_ball = self.model_ball.predict(
                im, 
                conf=0.25, 
                iou=0.45, 
                max_det=1000,
                imgsz=self.input_shape,
                verbose=False
            )
            
            boxes = results[0].boxes.cpu().numpy() if len(results) > 0 else None
            boxes_ball = results_ball[0].boxes.cpu().numpy() if len(results_ball) > 0 else None
            #整合
            numpy_like = self._merge_detection_results(boxes, boxes_ball)
            self.publisher.publish(numpy_like, 0.01)

    def _merge_detection_results(self, boxes, boxes_ball):
        if boxes is not None and len(boxes) > 0:
            main_detections = np.zeros((len(boxes), 6), dtype=np.float32)
            main_detections[:, :4] = boxes.xyxy  # 边界框坐标
            main_detections[:, 4] = boxes.conf   # 置信度
            main_detections[:, 5] = boxes.cls    # 类别
            mask = boxes.cls == 1
            main_detections[mask, 5] = 13
        else:
            main_detections = np.zeros((0, 6), dtype=np.float32)
            
        if boxes_ball is not None and len(boxes_ball) > 0:
            ball_detections = np.zeros((len(boxes_ball), 6), dtype=np.float32)
            ball_detections[:, :4] = boxes_ball.xyxy  # 边界框坐标
            ball_detections[:, 4] = boxes_ball.conf   # 置信度
            ball_detections[:, 5] = boxes_ball.cls + 1    # 类别
        else:
            ball_detections = np.zeros((0, 6), dtype=np.float32)

        merged_detections = np.vstack([main_detections, ball_detections])
        return merged_detections

    def end(self):
        self.stop = True


if __name__ == "__main__":
    t = Detector()
    t.start()
    t.join()