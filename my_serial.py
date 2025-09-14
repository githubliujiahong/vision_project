import cv2

import Bridge
from PyQt5.QtCore import QThread
import toml, os
import serial, struct
import numpy as np

# 将一个十位数转为4个二进制八位数
def intToBytes(value):
    src = [0,0,0,0]
    src[0] = value & 0xFF
    src[1] = (value>>8) & 0xFF
    src[2] = (value>>16) & 0xFF
    src[3] = (value>>24) & 0xFF
    sum = 0
    for i in range(4):
        sum += src[i]
    return src, sum

# ？
def signed_int_to_8bit(int_num):
    if int_num < 0:
        if int_num < -127:
            raise Exception("cannot transform a number smaller than -127 into a int8")
    else:
        if int_num > 127:
            raise Exception("cannot transform a number bigger than 127 into a int8")
    return np.uint8(int_num)

def unsigned_int_to_8bit(int_num):
    if int_num < 0:
        if int_num < 0:
            raise Exception("cannot transform a number smaller than 0 into a uint8")
    else:
        if int_num > 255:
            raise Exception("cannot transform a number bigger than 255 into a uint8")
    return np.uint8(int_num)
# 高位去掉
def int_to_8bit(int_num):
    return int(int_num) & 0xFF

def data_trans(raw_data): # 我们收到电控时的解码
    theta = raw_data[1] 
    R = raw_data[2]
    if raw_data[2] >= 128:
        theta -= 180
        R -= 1
        R ^= 0XFF
    R = R * 35 / 127
    return [raw_data[0], R, theta]

class Serial(QThread):
    def __init__(self, debuger=None):
        super().__init__()


        ROOT = 	os.getcwd()
        self.config = toml.load(os.path.join(ROOT, "config", "serial.toml"))


        self.baudrate = self.config["baudrate"]
        self.port_name = self.config["port_name"]
        self.data_length = self.config["data_length"]
        config = toml.load(os.path.join(ROOT, "config/config.toml"))
        specific_config = config["specific_config"]
        PATH = os.path.join(ROOT, "config", specific_config)
        PATH = PATH.replace('/', os.sep).replace("\\", os.sep) # 确保路径字符串中的路径分隔符在不同操作系统下都是一致的，以便正确地处理路径。
        config = toml.load(PATH)
        self.arm_length = config["decision"]["arm_length"]

        self.position = [0, 0] # ?
        self.stop = False
        self.debuger = debuger
        self.status = 0x00

        while True:
            try:
                self.serial = serial.Serial(self.port_name, self.baudrate, timeout=0.01)
                break
            except:
                print("Fail to open serial!")
        
        self.subscriber = Bridge.Subscriber("send_data")
        self.publisher = Bridge.Publisher("rec_data")
        self.pub_log = Bridge.Publisher("log")

    def run(self):
        while not self.stop:
            try:
                while self.serial.in_waiting:  # 串口中有数据等待时
                    data = self.serial.read_all()  # 从串口读取数据，将字节转换成字符串并储存在data中
                    self.serial.flush()
                    status = 0  # 0x53, 0x5A, 0x48, 0x59
                    for index in range(len(data)):
                        match data[index]:
                            case 0x53:
                                status = 0x5A
                                continue
                            case 0x5A if status == 0x5A:
                                status = 0x48
                                continue
                            case 0x48 if status == 0x48:
                                status = 0x59
                                continue
                            case 0x59 if status == 0x59:
                                try:
                                    if data[index] == 0xFD:
                                        self.pub_log.publish("卡住了")
                                    if sum(data[index + 1:index + 1 + self.data_length]) & 0xFF == data[index + 1 + self.data_length]:
                                        self.publisher.publish(list(data[index + 1:index + 1 + self.data_length]))
                                        # print(f"{data[index + 1:index + 1 + self.data_length]}")
                                        if data[index + 1] != self.status:
                                            self.status = data[index + 1]
                                            # print(self.status)
                                        status = 0x00
                                        break
                                except:
                                    status = 0x00
                                    continue
                            case _:
                                status = 0x00
                                continue

                try:
                    cmd = self.subscriber.get_message(0.003) 
                except:
                    continue
                send_data = self.get_new_protocol(*cmd)
                print(f"send_data: {send_data}")
                for i in range(20):
                    self.serial.write(send_data)
                    cv2.waitKey(10)
            except Exception as err:
                self.pub_log.publish("串口断开，尝试重启串口")
                print(err)
                self.serial.close()
                while True:
                    try:
                        self.serial = serial.Serial(self.port_name, self.baudrate, timeout=0.01)
                        break
                    except:
                        self.pub_log.publish("重启失败，继续尝试")
                self.pub_log.publish("串口重启成功")

        self.serial.close()

    def end(self):
        self.stop = True

    def get_new_protocol(self, order, R, theta):
        theta_degree = (theta) / np.pi * 180 + 2
        print(f"raw:{theta_degree = }, {R = }")
        R_protocol = R / (self.arm_length+0.4) * 127
        # R_protocol = R / 33.7 * 127


        print(f"{R_protocol = }")
        if theta_degree < 0:
            theta_degree = int(theta_degree + 180)
            R_protocol = int(-R_protocol)  # 这里可能由于位数问题导致转成1byte时符号位丢失
        else:
            theta_degree = int(theta_degree)
            R_protocol = int(R_protocol)
        print(f"after protocol: {theta_degree = }, {R_protocol = }")
        theta_8bit = int_to_8bit(theta_degree)
        R_8bit = int_to_8bit(R_protocol)
        print(f"R, theta in 8 bit = {R_8bit, theta_8bit}")
        data = [0x53, 0x5A, 0x48, 0x59, order, theta_8bit, R_8bit]
        data.append((order + theta_8bit + R_8bit + 1) & 0xFF)
        print(data)
        return struct.pack("8B", *data)
    
    def get_old_protocol(slef, cmd):
        bytes1, sum1 = intToBytes(cmd[1])
        bytes2, sum2 = intToBytes(cmd[2])
        order = cmd[0]
        ch = 0x01
        len = 0x13
        check_sum = (0x59 + 0x48 + 0x5A + 0x53 + ch + sum1 + sum2 + len + order) & 0xFF
        send_data = [0x53, 0x5A, 0x48, 0x59, ch, len, 0x00, 0x00, 0x00, order, *bytes1, *bytes2, check_sum]
        send_data = struct.pack("19B", *send_data)