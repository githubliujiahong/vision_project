import sys, os
PKG_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PKG_PATH)
import numpy as np
import toml, time
from PyQt5.QtCore import QThread
import Bridge, Position_calculate, my_serial


num2type = ["cube", "ball", "ball", "ball", "ball", "ball", "ball", "box", "box", "box", "box", "box", "box", "big_box"]

 # 平均滤波器
class Mean_filter():
    def __init__(self, max_size = np.inf):
        self.__max_size = max_size
        self.__data = []

    def push(self, data):
        while len(self.__data) >= self.__max_size:
            del self.__data[0]  #删除第一个元素
        self.__data.append(data)

    def get_average(self):     #得到
        if len(self.__data) == 0:
            return 0
        return sum(self.__data) / len(self.__data)

class Post_processor(QThread):
    #set_AI_signal = pyqtSignal(bool) # 定义了一个布尔类型的信号

    def set_IA(self):
        self.use_IA = True

    def set_AI(self, use_AI):
        self.use_AI = use_AI 
    
    def __init__(self, debuger=None):
        super().__init__()

        ROOT = 	os.getcwd()
        config = toml.load(os.path.join(ROOT, "config/config.toml"))
        specific_config = config["specific_config"]
        PATH = os.path.join(ROOT, "config", specific_config)
        PATH = PATH.replace('/', os.sep).replace("\\", os.sep) 
        self.config = toml.load(PATH)

        position_config = self.config["position_calculate"]
        position_config['K'] = self.config["camera"]['K']
        position_config['D'] = self.config["camera"]['D']
        self.position_calculator = Position_calculate.Position_calculator(debuger)
        decision_config = self.config["decision"]
        decision_config["image_shape"] = self.config["camera"]["image_shape"]
        

        #self.use_AI = True  # 是否使用自动决定目标
        self.manual_target = None
        self.place_target = None
        self.rec_data = None  #  
        self.stop = False
        self.cmd_queue:list = []
        self.now_status = 0xFF
        self.last_send_time = time.time()
        self.debuger = debuger
        self.last_R = self.decision_maker.arm_length
        self.last_directioin = 0
        self.dead_angle = self.config["decision"]["dead_angle"] / 180 * np.pi
        self.series = False

        self.subscriber_detect = Bridge.Subscriber("detect_res") 
        self.subscriber_manual = Bridge.Subscriber("manual_target")
        self.subscriber_place = Bridge.Subscriber("double_target")
        self.subscriber_action = Bridge.Subscriber("action")
        self.subscriber_rec = Bridge.Subscriber("rec_data")
        self.publisher = Bridge.Publisher("send_data")
        self.publisher_log = Bridge.Publisher("log")

        # self.set_AI_signal.connect(self.set_AI)

    def run(self):
        """
        后处理逻辑
        """
        pass

    def end(self):
        """
        想想看这部分怎么改
        """
        pass