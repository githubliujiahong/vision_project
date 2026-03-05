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
        
        _ = self.model.predict(np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8), verbose=False)
        _ = self.model_ball.predict(np.zeros((self.input_shape[0], self.input_shape[1], 3), dtype=np.uint8), verbose=False)
        print("detector init finished!")

    def run(self):
        while not self.stop:
            try:
                now_image = self.subscriber.get_message(0.001)
            except:
                continue
                
            im = cv2.cvtColor(now_image, cv2.COLOR_BGR2RGB)
            

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

            numpy_like = self._merge_detection_results(boxes, boxes_ball)
            self.publisher.publish(numpy_like, 0.01)

    def _merge_detection_results(self, boxes, boxes_ball):
        """
        请填写，此处为第三空
        检测结果自己查阅资料
        """
        


    def end(self):
        self.stop = True


if __name__ == "__main__":
    t = Detector()
    t.start()
    t.join()