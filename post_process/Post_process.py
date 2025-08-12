import sys, os
PKG_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PKG_PATH)
import numpy as np
import toml, time
from PyQt5.QtCore import QThread
import Bridge, Position_calculate, Decision, my_serial
from Decision import Target

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
        PATH = PATH.replace('/', os.sep).replace("\\", os.sep) # 确保路径字符串中的路径分隔符在不同操作系统下都是一致的，以便正确地处理路径。
        self.config = toml.load(PATH)

        position_config = self.config["position_calculate"]
        position_config['K'] = self.config["camera"]['K']
        position_config['D'] = self.config["camera"]['D']
        self.position_calculator = Position_calculate.Position_calculator(debuger)
        decision_config = self.config["decision"]
        decision_config["image_shape"] = self.config["camera"]["image_shape"]
        self.decision_maker = Decision.Decision_maker(decision_config, debuger)

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
        while not self.stop:
            #  获取消息
            try:
                self.manual_target = self.subscriber_manual.get_message(0.003)
            except:
                pass
            try:
                self.place_target = self.subscriber_place.get_message(0.003)  
            except:
                pass
            recieve_detect = True
            try:
                now_detect = self.subscriber_detect.get_message(0.003) # 返回的是一个列表
            except:
                recieve_detect = False
            try:
                self.rec_data = self.subscriber_rec.get_message(0.003)
                self.now_status, R, theta = my_serial.data_trans(self.rec_data)
                if theta > 0:
                    self.last_directioin = 1
                    # print("theta > 0")
                elif theta < 0:
                    self.last_directioin = -1
                    # print("theta < 0")
                else:
                    self.last_directioin = 0
                    # print("theta == 0")
                if self.cmd_queue and self.now_status == self.cmd_queue[0][0]:
                    del self.cmd_queue[0]
                    self.publisher_log(f"任务 {self.now_status} 已在执行")
                    if not self.cmd_queue:
                        self.series = False
                if self.cmd_queue[0][0] == 0xFE and self.rec_data[0] == 0xFF:
                    del self.cmd_queue[0]
                    self.publisher_log(f"成功重置电控状态")
            except:
                pass
  
            targets = []
            if recieve_detect:
                # 逐个计算坐标
                for target in now_detect:
                    pix_pt = np.array([(target[0] + target[2]) / 2, (target[1] + target[3]) / 2])
                    if int(target[5]) == 13:
                        base_pt = self.position_calculator.bigbox_get_base(target[:4])
                    else:
                        base_pt = self.position_calculator.pix2base(pix_pt, num2type[int(target[5])])
                    targets.append(Target(target, base_pt))
            
                # 更新决策器
                self.decision_maker.update(targets)

            try:
                # action定义集：1：抓球；2：抓魔方
                now_action = self.subscriber_action.get_message(0.003) 
            except:
                now_action = None
            if now_action:
                # 更新指令列表
                # 逆时针转90°、转180°、顺时针转90°
                if now_action != 5:
                    self.series = False
                match now_action:
                    case -1:
                        now_cmd = [[0x00, self.last_R, -np.pi / 2]]
                        self.publisher_log.publish("制定任务：逆时针转90°")
                    case -2:
                        now_cmd = [[0x00, self.last_R, np.pi]]
                        self.publisher_log.publish("制定任务：转180°")
                    case -3:
                        now_cmd = [[0x00, self.last_R, np.pi / 2]]
                        self.publisher_log.publish("制定任务：顺时针转90°")
                    case -4:
                        now_cmd = []
                        self.cmd_queue.clear()
                        self.publisher_log.publish("任务清空")
                    case -5:
                        now_cmd = [[0xFE, self.last_R, 0]]
                        self.publisher_log.publish("清空电控状态")
                    case -6:
                        now_cmd = [[0x00, 0, 0]]
                        self.publisher_log.publish("制定任务：回到原点")
                    case -7:
                        now_cmd = [[0x00, self.decision_maker.arm_length, 0]]
                        self.publisher_log.publish("制定任务：移开爪")
                    case 14:
                        now_cmd = [[0xFC, self.decision_maker.arm_length, 0]]
                        self.publisher_log.publish("制定任务：电控复位")
                    case _:
                            match now_action:
                                case 1:
                                    try:
                                        now_cmd = [[0x01, *self.position_calculator.base2polar(self.decision_maker.get_ball_base())]]
                                        self.publisher_log.publish("制定任务：抓球")
                                        print("catch ball")
                                    except:
                                        self.publisher_log.publish("未识别到球")
                                        now_cmd = []
                                case 2:
                                    try:
                                        now_cmd = [[0x02, *self.position_calculator.base2polar(self.decision_maker.get_cube_base())]]
                                        self.publisher_log.publish("制定任务：抓魔方")
                                        print("catch cube")
                                    except:
                                        self.publisher_log.publish("未识别到魔方")
                                        now_cmd = []
                                case 3:  # 放球
                                    try:
                                        now_cmd = [[0x03, *self.position_calculator.base2polar(self.decision_maker.get_box_base())]]
                                        self.publisher_log.publish("制定任务：放球")
                                        print("throw ball")
                                    except:
                                        self.publisher_log.publish("未识别到盒子")
                                        now_cmd = []
                                case 4:  # 放魔方
                                    try:
                                        if np.linalg.norm(self.decision_maker.get_big_base()) > self.position_calculator.bigbox_R + self.decision_maker.arm_length - 5:
                                            self.publisher_log.publish("大盘过远")
                                            now_cmd = []
                                        else:
                                            big_polar = self.position_calculator.base2polar(self.decision_maker.get_big_base())
                                            big_polar[0] = min(big_polar[0], self.decision_maker.arm_length)
                                            now_cmd = [[0x04, *big_polar]]
                                            self.publisher_log.publish("制定任务：放魔方")
                                    except:
                                        self.publisher_log.publish("未识别到大盘")
                                        now_cmd = []
                                case 5:  # 扫堂腿
                                    if not self.manual_target:
                                        self.publisher_log.publish("未指定终点")
                                        now_cmd = []
                                    if not self.place_target:
                                        self.publisher_log.publish("未指定落爪点")
                                        now_cmd = []
                                        
                                    else:
                                        self.publisher_log.publish("制定任务：横扫")
                                        final_target = self.position_calculator.base2polar(self.position_calculator.pix2base(self.manual_target, "cube"))
                                        open_target = self.position_calculator.base2polar(self.position_calculator.pix2base(self.place_target, "cube"))
                                        delta_theta = final_target[1] - open_target[1]
                                        if delta_theta > np.pi:
                                            delta_theta -= 2 * np.pi
                                        elif delta_theta < -np.pi:
                                            delta_theta += 2 * np.pi
                                        now_cmd = [[0x00, *open_target]]
                                        now_cmd += [[0x0C, open_target[0], 0]]
                                        now_cmd += [[0x00, final_target[0], delta_theta]]
                                        now_cmd += [[0x07, final_target[0], 0]]
                                        self.series = True
                                case 6:  # 复位
                                    self.publisher_log.publish("制定任务：复位")
                                    if not self.rec_data is None:
                                        print(self.rec_data)
                                        now_cmd = [[0x06, self.rec_data[2], 0]] 
                                    else:
                                        now_cmd = [[0x06, 28, 0]]
                                case 7:  # 抓球组合技
                                    try:
                                        now_cmd = [[0x09, *self.position_calculator.base2polar(self.decision_maker.get_ball_base())]]
                                        self.publisher_log.publish("制定任务：抓边沿球")
                                    except:
                                        self.publisher_log.publish("未识别到球")
                                        now_cmd = []
                                case 8:  # 抓魔方组合技
                                    try:
                                        now_cmd = [[0x0A, *self.position_calculator.base2polar(self.decision_maker.get_cube_base())]]
                                        self.publisher_log.publish("制定任务：抓边沿魔方")
                                    except:
                                        self.publisher_log.publish("未识别到魔方")
                                        now_cmd = []
                                case 13:  # 上提机械臂
                                        self.publisher_log.publish("制定任务：上提")
                                        if not self.rec_data is None:
                                            now_cmd = [[0x0B, self.rec_data[1], 0]] 
                                        else:
                                            now_cmd = [[0x0B, 28, 0]]  #  新加的协议
                                case _:
                                    if not self.manual_target:
                                        self.publisher_log.publish("未设置目标") 
                                        now_cmd = [] 
                                    else:
                                        match now_action:
                                            case 9:  # 抓球
                                                self.publisher_log.publish("制定任务：手动抓球")
                                                print("catch ball")
                                                action = 0x01
                                                type = "ball"
                                            case 10:  # 抓魔方
                                                    self.publisher_log.publish("制定任务：手动抓魔方")
                                                    print("catch cube")
                                                    action = 0x02
                                                    type = "cube"
                                            case 11:  # 放球
                                                self.publisher_log.publish("制定任务：手动放球")
                                                print("throw ball")
                                                action = 0x03
                                                type = "box"
                                            case 12:  # 放魔方
                                                self.publisher_log.publish("制定任务：手动放魔方")
                                                print("throw cube")
                                                action = 0x04
                                                type = "floor"
                                            case 15:
                                                self.publisher_log.publish("制定任务：手动抓边沿球")
                                                action = 0x09
                                                type = "ball"
                                            case 16:
                                                self.publisher_log.publish("制定任务：手动抓边沿魔方")
                                                action = 0x0A
                                                type = "cube"
                                            case _:
                                                self.publisher_log.publish("未知指令")
                                                now_cmd = []
                                        now_cmd = [[action, *self.position_calculator.base2polar(self.position_calculator.pix2base(self.manual_target, type))]]
                                        print("R, theta in pc:\n", self.position_calculator.base2polar(self.position_calculator.pix2base(self.manual_target, type)))
                if now_cmd:
                    invalid = False
                    for single_cmd in now_cmd:
                        if single_cmd and single_cmd[1] >= self.decision_maker.arm_length + 2:  # 这里不用arm_length作为阈值主要是有时候目标超限一点点也是能抓到的，只要指令不溢出就行
                            print("TOO FAR TO REACH!!!")
                            self.publisher_log.publish("距离过远无法抓取") 
                            invalid = True
                    if not invalid:
                        for i in range(len(now_cmd)):
                            # 这一步认为绕y轴z轴的旋转相对于绕x轴的旋转是一个小量
                            now_cmd[i][1] = min(now_cmd[i][1], self.decision_maker.arm_length)
                        self.cmd_queue = now_cmd
                    print(f"{now_cmd = }")
            if not self.cmd_queue:
                continue
            if self.series:
                if time.time() - self.last_send_time > 0.1:
                    self.last_R = self.cmd_queue[0][1]
                    cmd, R, theta = self.cmd_queue[0]
                    if theta * self.last_directioin < 0:
                        if theta < 0:
                            theta -= self.dead_angle
                        elif theta > 0:
                            theta += self.dead_angle
                    self.publisher.publish([cmd, R, theta])
                    self.publisher_log.publish(f"发送指令{self.cmd_queue[0]}")
                    self.last_send_time = time.time()
            else:
                self.last_R = self.cmd_queue[0][1]
                cmd, R, theta = self.cmd_queue[0]
                if theta * self.last_directioin < 0:
                    if theta < 0:
                        theta -= self.dead_angle
                    elif theta > 0:
                        theta += self.dead_angle
                self.publisher.publish([cmd, R, theta])
                self.publisher_log.publish(f"发送指令{self.cmd_queue[0]}")
                self.last_send_time = time.time()
                self.cmd_queue.clear()

    def end(self):
        self.stop = True