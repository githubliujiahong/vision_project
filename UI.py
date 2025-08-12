from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QMainWindow, QLabel, QPushButton, QGroupBox, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2, toml, Bridge, os
from post_process import Position_calculate
import numpy as np

class UI_thread(QThread):
    update_image_signal = pyqtSignal(list)
    update_log_signal = pyqtSignal(str)
    update_rec_signal = pyqtSignal(list)
    update_icon_signal = pyqtSignal(list)
    update_color_signal = pyqtSignal(np.ndarray)
    def __init__(self, show_icon):
        super().__init__()

        self.subscriber = Bridge.Subscriber("image")  # 设置一个叫“image”的订阅者
        self.detect_sub = Bridge.Subscriber("detect_res")
        self.ball_sub = Bridge.Subscriber("ball_pix")
        self.cube_sub = Bridge.Subscriber("cube_pix")
        self.log_sub = Bridge.Subscriber("log", 10)
        self.rec_sub = Bridge.Subscriber("rec_data")
        if show_icon:
            self.icon_sub = Bridge.Subscriber("icon")
        self.stop = False
        self.detect_res = None
        self.ball_pix = None
        self.cube_pix = None
        self.show_icon = show_icon

    def run(self):
        while not self.stop:
            try:
                now_image = self.subscriber.get_message(0.05)
            except:
                # print("Mainwindow fail to sub image!")
                continue
            try:
                self.detect_res = self.detect_sub.get_message(0.003)
                self.update_color_signal.emit(self.detect_res)
            except:
                pass
            try:
                self.ball_pix = self.ball_sub.get_message(0.003)
                self.ball_pix = [int(self.ball_pix[0]), int(self.ball_pix[1])]
                # print(f"{self.ball_pix = }")
            except:
                pass
            try:
                self.cube_pix = self.cube_sub.get_message(0.003)
                self.cube_pix = [int(self.cube_pix[0]), int(self.cube_pix[1])]
                # print(f"{self.cube_pix = }")
            except:
                pass
            try:
                log = ""
                while not self.log_sub.is_empty():
                    log += self.log_sub.get_message(0.003) + '\n'
                self.update_log_signal.emit(log)
            except:
                pass
            try:
                rec_data = self.rec_sub.get_message(0.003)
                self.update_rec_signal.emit(rec_data) 
            except:
                pass
            if self.show_icon:
                try:
                    icon = self.icon_sub.get_message(0.003)
                    self.update_icon_signal.emit([icon])
                except:
                    pass
            self.update_image_signal.emit([now_image, self.cube_pix, self.ball_pix, self.detect_res])
            
            
    def end(self):
        self.stop = True

class Main_window(QMainWindow):
    restart_camera = pyqtSignal()
    def __init__(self, post_process_th, debuger):
        super().__init__()

        ROOT = 	os.getcwd()
        self.config = toml.load(os.path.join(ROOT, "config/UI.toml"))
        self.show_icon = self.config["show_icon"]
        self.color_his_path = self.config["color_his_path"]
        self.reshape_ratio = self.config["reshape_ratio"]
        # 新增：当前缩放比例，全屏时会调整
        self.current_reshape_ratio = self.reshape_ratio
        
        config = toml.load(os.path.join(ROOT, "config/config.toml"))
        specific_config = config["specific_config"]
        PATH = os.path.join(ROOT, "config", specific_config)
        PATH = PATH.replace('/', os.sep).replace("\\", os.sep) # 确保路径字符串中的路径分隔符在不同操作系统下都是一致的，以便正确地处理路径。
        config = toml.load(PATH)
        K = np.array(config["camera"]["K"], dtype=np.float32)
        D = np.array(config["camera"]["D"], dtype=np.float32)
        alpha = config["camera"]["alpha"]
        self.image_size = config["camera"]["image_shape"]
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, self.image_size, alpha, self.image_size, True)
        self.camera_centre = np.array([int(new_K[0, 2]), int(new_K[1, 2])])
        self.arm_length = config["decision"]["arm_length"]
        
        self.post_process_th = post_process_th
        self.debuger = debuger
        if not debuger == None:
            self.debuger.add_param("manual_R", 1)
            self.debuger.add_param("ball_R", 10)
            self.debuger.add_param("cube_R", 10)
            self.debuger.add_param("box_R", 10)
        self.use_AI = True  # 是否使用自动决定目标
        self.names = ["cube", "yellow", "green", "brown", "blue", "pink", "black",
                      "yellow_box", "green_box", "brown_box", "blue_box", "pink_box", "black_box", "big_box"]
        self.click_point = None
        self.double_click_point = None
        self.normalization = False
        self.box_color_queue = []
        self.box_color_text = ""
        self.auto_record_color = False
        self.R = 0

        try:
            tem = np.loadtxt(self.color_his_path, dtype=str) ##转字符串
            self.color_his = tem.tolist()
            # 判断
            for color in self.color_his:
                if color not in ["white", "black", "pink", "brown", "blue", "yellow", "green"]:
                    raise Exception(f"unexpected value: {color}")
            print("find color history")
        except Exception as err:
            print(f"error: {err}")
            self.color_his = np.array(["white"] * 6)
        print(self.color_his)

        self.pc = Position_calculate.Position_calculator(None)
        self.origin_pix = np.array(self.pc.get_base_origin_in_pix("floor"), dtype=np.uint)
        # self.origin_pix_ball = np.array(self.pc.get_base_origin_in_pix("ball"), dtype=np.uint)
        # pt1 = self.pc.base2pix(self.pc.polar2base([self.arm_length, 0], "ball"))
        # pt2 = self.pc.base2pix(self.pc.polar2base([self.arm_length, np.pi / 2], "ball"))
        # self.axes1 = (int(np.linalg.norm(pt1 - self.origin_pix_ball)), int(np.linalg.norm(pt2 - self.origin_pix_ball)))

        # self.origin_pix_box = np.array(self.pc.get_base_origin_in_pix("box"), dtype=np.uint)
        # pt1 = self.pc.base2pix(self.pc.polar2base([self.arm_length, 0], "box"))
        # pt2 = self.pc.base2pix(self.pc.polar2base([self.arm_length, np.pi / 2], "box"))
        # self.axes2 = (int(np.linalg.norm(pt1 - self.origin_pix_box)), int(np.linalg.norm(pt2 - self.origin_pix_box)))
        # resized_shape = np.array(np.array(self.image_size) * self.reshape_ratio, dtype=np.uint16)
        self.ball_pts = self.pc.get_rotation_elliptic_pts("ball", self.image_size)
        self.box_pts = self.pc.get_rotation_elliptic_pts("box", self.image_size)

        # 又臭又长的UI初始化
        self.setWindowTitle("MAN!!! What Can I Say?")  # 窗口标题的设置

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.image_label = QLabel()
        height, width = self.image_size
        bytes_per_line = 3 * width
        q_image = QImage(np.zeros((height, width, 3), dtype=np.uint8), width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        # 图片显示
        layout = QHBoxLayout()
        image_layout = QVBoxLayout()
        tool_layout = QHBoxLayout()

        if self.show_icon:
            self.icon_image = QLabel()
            height, width = [100, 100]
            bytes_per_line = 3 * width
            q_image = QImage(np.zeros((100, 100, 3), dtype=np.uint8), width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.icon_image.setPixmap(pixmap)
            tool_layout.addWidget(self.icon_image)

        self.restart_buttom = QPushButton()
        self.restart_buttom.resize(self.restart_buttom.sizeHint())
        self.restart_buttom.setText("重启相机")
        self.restart_buttom.clicked.connect(self.restart_camera_callback)
        tool_layout.addWidget(self.restart_buttom)
        image_layout.addLayout(tool_layout)
        image_layout.addWidget(self.image_label)

        layout.addLayout(image_layout)

        self.image_label.mousePressEvent = self.on_label_click

        # 工作区显示
        workspace_layout = QVBoxLayout()

        # 颜色检测记录
        color_his_layout = QHBoxLayout()
        self.detect_color_show = QPushButton()
        self.detect_color_show.setStyleSheet("background-color: white;")
        self.detect_color_show.setFixedSize(100, 100)
        color_his_layout.addWidget(self.detect_color_show)

        color_his_right_layout = QVBoxLayout()
        self.auto_record_button = QPushButton()
        self.auto_record_button.resize(self.auto_record_button.sizeHint())
        self.auto_record_button.setText('记录颜色：手动')
        self.auto_record_button.setStyleSheet('color: black;')
        self.auto_record_button.clicked.connect(self.change_record_color_status_callback)
        color_his_right_layout.addWidget(self.auto_record_button)

        self.color_his_text = QTextEdit()
        self.color_his_text.setReadOnly(True)
        color_his_right_layout.addWidget(self.color_his_text)

        color_his_layout.addLayout(color_his_right_layout)

        # 颜色选记录
        color_select_layout = QHBoxLayout()
        yellow_button = QPushButton()
        yellow_button.setStyleSheet("background-color: yellow")
        yellow_button.setFixedWidth(50)  
        yellow_button.setFixedHeight(100)
        yellow_button.clicked.connect(self.set_yellow)
        color_select_layout.addWidget(yellow_button)
        
        bule_button = QPushButton()
        bule_button.setStyleSheet("background-color: blue")
        bule_button.setFixedWidth(50)  
        bule_button.setFixedHeight(100)
        bule_button.clicked.connect(self.set_blue)
        color_select_layout.addWidget(bule_button)
        
        pink_button = QPushButton()
        pink_button.setStyleSheet("background-color: pink")
        pink_button.setFixedWidth(50)  
        pink_button.setFixedHeight(100)
        pink_button.clicked.connect(self.set_pink)
        color_select_layout.addWidget(pink_button)
        
        black_button = QPushButton()
        black_button.setStyleSheet("background-color: black")
        black_button.setFixedWidth(50)  
        black_button.setFixedHeight(100)
        black_button.clicked.connect(self.set_black)
        color_select_layout.addWidget(black_button)
        
        green_button = QPushButton()
        green_button.setStyleSheet("background-color: green")
        green_button.setFixedWidth(50)  
        green_button.setFixedHeight(100)
        green_button.clicked.connect(self.set_green)
        color_select_layout.addWidget(green_button)
        
        brown_button = QPushButton()
        brown_button.setStyleSheet("background-color: brown")
        brown_button.setFixedWidth(50)  
        brown_button.setFixedHeight(100)
        brown_button.clicked.connect(self.set_brown)
        color_select_layout.addWidget(brown_button)
        
        # 记录颜色的要求
        self.click_color = "white"
        color_layout = QHBoxLayout()
        color_group_box = QGroupBox("小球颜色")
        color_layout_inner = QVBoxLayout()

        self.clear_color_button = QPushButton()
        self.clear_color_button.setText("清空颜色记录")
        self.clear_color_button.setStyleSheet('color: red;')
        self.clear_color_button.clicked.connect(self.clear_color_callback)
        color_layout_inner.addWidget(self.clear_color_button)

        ball_first_layout = QHBoxLayout()
        ball_second_layout = QHBoxLayout()
        self.ball_box_button_1 = ColorButton(self, "1")
        self.ball_box_button_1.clicked.connect(self.ball_box_button_1.set_selected_color)
        if self.color_his[0] == "black":
            self.ball_box_button_1.setStyleSheet(f"color: white; background-color: {self.color_his[0]}")
        else:
            self.ball_box_button_1.setStyleSheet(f"color: black; background-color: {self.color_his[0]}")
        self.ball_box_button_1.update_color_signal.connect(self.update_color)
        self.ball_box_button_2 = ColorButton(self, "2")
        self.ball_box_button_2.clicked.connect(self.ball_box_button_2.set_selected_color)
        if self.color_his[1] == "black":
            self.ball_box_button_2.setStyleSheet(f"color: white; background-color: {self.color_his[1]}")
        else:
            self.ball_box_button_2.setStyleSheet(f"color: black; background-color: {self.color_his[1]}")
        self.ball_box_button_2.update_color_signal.connect(self.update_color)
        self.ball_box_button_3 = ColorButton(self, "3")
        self.ball_box_button_3.clicked.connect(self.ball_box_button_3.set_selected_color)
        if self.color_his[2] == "black":
            self.ball_box_button_3.setStyleSheet(f"color: white; background-color: {self.color_his[2]}")
        else:
            self.ball_box_button_3.setStyleSheet(f"color: black; background-color: {self.color_his[2]}")
        self.ball_box_button_3.update_color_signal.connect(self.update_color)
        ball_first_layout.addWidget(self.ball_box_button_1)
        ball_first_layout.addWidget(self.ball_box_button_2)
        ball_first_layout.addWidget(self.ball_box_button_3)

        self.ball_box_button_4 = ColorButton(self, "4")
        self.ball_box_button_4.clicked.connect(self.ball_box_button_4.set_selected_color)
        if self.color_his[3] == "black":
            self.ball_box_button_4.setStyleSheet(f"color: white; background-color: {self.color_his[3]}")
        else:
            self.ball_box_button_4.setStyleSheet(f"color: black; background-color: {self.color_his[3]}")
        self.ball_box_button_4.update_color_signal.connect(self.update_color)
        self.ball_box_button_5 = ColorButton(self, "5")
        self.ball_box_button_5.clicked.connect(self.ball_box_button_5.set_selected_color)
        if self.color_his[4] == "black":
            self.ball_box_button_5.setStyleSheet(f"color: white; background-color: {self.color_his[4]}")
        else:
            self.ball_box_button_5.setStyleSheet(f"color: black; background-color: {self.color_his[4]}")
        self.ball_box_button_5.update_color_signal.connect(self.update_color)
        self.ball_box_button_6 = ColorButton(self, "6")
        self.ball_box_button_6.clicked.connect(self.ball_box_button_6.set_selected_color)
        if self.color_his[5] == "black":
            self.ball_box_button_6.setStyleSheet(f"color: white; background-color: {self.color_his[5]}")
        else:
            self.ball_box_button_6.setStyleSheet(f"color: black; background-color: {self.color_his[5]}")
        self.ball_box_button_6.update_color_signal.connect(self.update_color)
        ball_second_layout.addWidget(self.ball_box_button_4)
        ball_second_layout.addWidget(self.ball_box_button_5)
        ball_second_layout.addWidget(self.ball_box_button_6)

        color_layout_inner.addLayout(ball_first_layout)
        color_layout_inner.addLayout(ball_second_layout)
        color_group_box.setLayout(color_layout_inner)
        color_group_box.setMaximumHeight(200)
        color_group_box.setMaximumWidth(400)
        color_layout.addWidget(color_group_box) 

        # 自动夹爪按钮布局
        AI_paw_layout = QHBoxLayout()
        AI_paw_group_box = QGroupBox("自动模式")
        AI_paw_layout_inner = QVBoxLayout()

        AI_ball_layout = QHBoxLayout()
        self.AI_catch_ball_buttom = QPushButton("抓取球")
        self.AI_catch_ball_buttom.clicked.connect(self.AI_catch_ball_callback)
        self.AI_throw_ball_buttom = QPushButton("放下球")
        self.AI_throw_ball_buttom.clicked.connect(self.AI_throw_ball_callback)
        self.AI_catch_ball_buttom.resize(self.AI_catch_ball_buttom.sizeHint())
        self.AI_throw_ball_buttom.resize(self.AI_throw_ball_buttom.sizeHint())
        AI_ball_layout.addWidget(self.AI_catch_ball_buttom)
        AI_ball_layout.addWidget(self.AI_throw_ball_buttom)

        AI_cube_layout = QHBoxLayout()
        self.AI_catch_cube_buttom = QPushButton("抓取魔方")
        self.AI_catch_cube_buttom.clicked.connect(self.AI_catch_cube_callback)
        self.AI_throw_cube_buttom = QPushButton("放下魔方")
        self.AI_throw_cube_buttom.clicked.connect(self.AI_throw_cube_callback)
        self.AI_catch_cube_buttom.resize(self.AI_catch_cube_buttom.sizeHint())
        self.AI_throw_cube_buttom.resize(self.AI_throw_cube_buttom.sizeHint())
        AI_cube_layout.addWidget(self.AI_catch_cube_buttom)
        AI_cube_layout.addWidget(self.AI_throw_cube_buttom)

        self.AI_edge_ball_buttom = QPushButton("抓边沿球")
        self.AI_edge_ball_buttom.clicked.connect(self.AI_edge_ball_callback)
        self.AI_edge_ball_buttom.resize(self.AI_edge_ball_buttom.sizeHint())

        AI_paw_layout_inner.addLayout(AI_ball_layout)
        AI_paw_layout_inner.addLayout(AI_cube_layout)
        AI_paw_layout_inner.addWidget(self.AI_edge_ball_buttom)
        AI_paw_group_box.setLayout(AI_paw_layout_inner)
        AI_paw_group_box.setMaximumHeight(200)
        AI_paw_group_box.setMaximumWidth(250)
        AI_paw_layout.addWidget(AI_paw_group_box)

        paw_layout = QHBoxLayout()
        paw_group_box = QGroupBox("手动模式")
        paw_layout_inner = QVBoxLayout()

        ball_layout = QHBoxLayout()
        self.catch_ball_buttom = QPushButton("抓取球")
        self.catch_ball_buttom.clicked.connect(self.catch_ball_callback)
        self.throw_ball_buttom = QPushButton("放下球")
        self.throw_ball_buttom.clicked.connect(self.throw_ball_callback)
        self.catch_ball_buttom.resize(self.catch_ball_buttom.sizeHint())
        self.throw_ball_buttom.resize(self.throw_ball_buttom.sizeHint())
        ball_layout.addWidget(self.catch_ball_buttom)
        ball_layout.addWidget(self.throw_ball_buttom)

        cube_layout = QHBoxLayout()
        self.catch_cube_buttom = QPushButton("抓取魔方")
        self.catch_cube_buttom.clicked.connect(self.catch_cube_callback)
        self.throw_cube_buttom = QPushButton("放下魔方")
        self.throw_cube_buttom.clicked.connect(self.throw_cube_callback)
        self.catch_cube_buttom.resize(self.catch_cube_buttom.sizeHint())
        self.throw_cube_buttom.resize(self.throw_cube_buttom.sizeHint())
        cube_layout.addWidget(self.catch_cube_buttom)
        cube_layout.addWidget(self.throw_cube_buttom)

        self.edge_ball_buttom = QPushButton("抓边沿球")
        self.edge_ball_buttom.clicked.connect(self.edge_ball_callback)
        self.edge_ball_buttom.resize(self.edge_ball_buttom.sizeHint())

        paw_layout_inner.addLayout(ball_layout)
        paw_layout_inner.addLayout(cube_layout)
        paw_layout_inner.addWidget(self.edge_ball_buttom)
        paw_group_box.setLayout(paw_layout_inner)
        paw_group_box.setMaximumHeight(200)
        paw_group_box.setMaximumWidth(250)
        paw_layout.addWidget(paw_group_box)

        fancy_trick_layout_inner = QVBoxLayout()
        fancy_trick_groupbox = QGroupBox()
        fancy_trick_groupbox.setTitle("花活")
        
        self.release_button = QPushButton("原地松爪")
        self.release_button.clicked.connect(self.loose_callback)
        self.release_button.resize(self.release_button.sizeHint())
        fancy_trick_layout_inner.addWidget(self.release_button)

        self.move_cube_buttom = QPushButton("四两拨千斤")
        self.move_cube_buttom.clicked.connect(self.move_cube_callback)
        self.move_cube_buttom.resize(self.move_cube_buttom.sizeHint())
        fancy_trick_layout_inner.addWidget(self.move_cube_buttom)

        self.raise_arm_button = QPushButton("舵臂上提")
        self.raise_arm_button.clicked.connect(self.raise_arm_callback)
        self.raise_arm_button.resize(self.raise_arm_button.sizeHint())
        fancy_trick_layout_inner.addWidget(self.raise_arm_button)
        fancy_trick_layout = QHBoxLayout()
        fancy_trick_layout.addWidget(fancy_trick_groupbox)

        fancy_trick_groupbox.setLayout(fancy_trick_layout_inner)
        fancy_trick_groupbox.setMaximumHeight(200)
        fancy_trick_groupbox.setMaximumWidth(250)
        
        workspace_layout.addLayout(AI_paw_layout)
        workspace_layout.addLayout(paw_layout)
        workspace_layout.addLayout(fancy_trick_layout)
        workspace_layout.addLayout(color_his_layout)
        workspace_layout.addLayout(color_layout)
        workspace_layout.addLayout(color_select_layout)
        layout.addLayout(workspace_layout)

        # 状态查看区
        status_layout = QVBoxLayout()

        move_box = QGroupBox()
        move_box.setTitle("调整爪位置")
        move_layout_inner = QHBoxLayout()
        turn_around_layout = QVBoxLayout()
        line1 = QHBoxLayout()
        self.counterclockwise_90 = QPushButton()
        self.counterclockwise_90.setText("逆时针旋转90°")
        self.counterclockwise_90.resize(self.counterclockwise_90.sizeHint())
        self.counterclockwise_90.clicked.connect(self.counterclockwise_90_callback)
        self.clockwise_90 = QPushButton()
        self.clockwise_90.setText("顺时针旋转90°")
        self.clockwise_90.resize(self.clockwise_90.sizeHint())
        self.clockwise_90.clicked.connect(self.clockwise_90_callback)
        line1.addWidget(self.counterclockwise_90)
        line1.addWidget(self.clockwise_90)
        turn_around_layout.addLayout(line1)

        self.turn_180 = QPushButton()
        self.turn_180.setText("旋转180°")
        self.turn_180.resize(self.turn_180.sizeHint())
        self.turn_180.clicked.connect(self.turn_180_callback)
        turn_around_layout.addWidget(self.turn_180)

        self.return_origin = QPushButton()
        self.return_origin.setText("回到原点")
        self.return_origin.resize(self.return_origin.sizeHint())
        self.return_origin.clicked.connect(self.return_origin_callback)
        turn_around_layout.addWidget(self.return_origin)

        self.go_end = QPushButton()
        self.go_end.setText("移开抓")
        self.go_end.resize(self.go_end.sizeHint())
        self.go_end.clicked.connect(self.go_end_callback)
        turn_around_layout.addWidget(self.go_end)

        self.reset_button = QPushButton()
        self.reset_button.setText("电控复位")
        self.reset_button.resize(self.reset_button.sizeHint())
        self.reset_button.clicked.connect(self.reset_callback)
        turn_around_layout.addWidget(self.reset_button)

        move_layout_inner.addLayout(turn_around_layout)
        move_box.setLayout(move_layout_inner)
        status_layout.addWidget(move_box)

        # 状态区
        self.status = QLabel()
        self.status.setText("电控当前状态：null\nR：null\ntheta：null")
        status_layout.addWidget(self.status)
        self.log_title = QLabel()
        self.log_title.setText("日志")
        status_layout.addWidget(self.log_title)

        self.clear_buttom = QPushButton()
        self.clear_buttom.setText("清空日志")
        self.clear_buttom.resize(self.clear_buttom.sizeHint())
        self.clear_buttom.clicked.connect(self.clear_callback)
        status_layout.addWidget(self.clear_buttom)

        self.stop_task_buttom = QPushButton()
        self.stop_task_buttom.setText("停止任务")
        self.stop_task_buttom.resize(self.stop_task_buttom.sizeHint())
        self.stop_task_buttom.clicked.connect(self.stop_task_callback)
        status_layout.addWidget(self.stop_task_buttom)

        self.clear_singlechip_buttom = QPushButton()
        self.clear_singlechip_buttom.setText("清空单片机指令")
        self.clear_singlechip_buttom.resize(self.clear_singlechip_buttom.sizeHint())
        self.clear_singlechip_buttom.clicked.connect(self.clear_singlechip_callback)
        status_layout.addWidget(self.clear_singlechip_buttom)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumWidth(300)
        status_layout.addWidget(self.log_widget)
        layout.addLayout(status_layout)
        
        # 设置布局拉伸权重，确保图像区域在全屏时可扩展
        layout.setStretch(0, 3)  # 图像区域权重最高
        layout.setStretch(1, 1)  # 工作区权重
        layout.setStretch(2, 1)  # 状态区权重
        
        self.central_widget.setLayout(layout)

        self.publisher_target = Bridge.Publisher("manual_target")
        self.publisher_dou_tar = Bridge.Publisher("double_target")
        self.publisher_action = Bridge.Publisher("action")  # 抓球:0；抓魔方:1；放球:2；放魔方:3

        self.ui_thread = UI_thread(self.show_icon)
        self.ui_thread.update_image_signal.connect(self.update_image)
        self.ui_thread.update_log_signal.connect(self.update_log)
        self.ui_thread.update_rec_signal.connect(self.update_rec)
        self.ui_thread.update_color_signal.connect(self.update_box_color)
        if self.show_icon:
            self.ui_thread.update_icon_signal.connect(self.update_icon)

    def start(self):
        self.ui_thread.start()

    def end(self):
        self.ui_thread.end()

    def wait(self):
        self.ui_thread.wait()

    def update_image(self, bag):
        image, cube_pix, ball_pix, detect_res = bag
        box_pix = None
        if self.normalization:
            for i in range(3):
                channel = image[:, :, i]
                image[:, :, i] = cv2.equalizeHist(channel)
        # 绘制识别框
        if not detect_res is None:
            if len(detect_res):
                for det in detect_res:
                    *xyxy, confidence, cls = det
                    pt1 = [int(xyxy[0]), int(xyxy[1])]
                    pt2 = [int(xyxy[2]), int(xyxy[3])]
                    if int(cls) > 13 or int(cls) < 0:
                        label = f"unknown {confidence:.2f}"
                        image = cv2.rectangle(image, pt1, pt2, (0, 0, 255), 1)
                        image = cv2.putText(image, label, [pt1[0] + 10, pt1[1] + 10], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        image = cv2.rectangle(image, pt1, pt2, self.get_color(int(cls)), 1)
                        if int(cls) > 6:
                            label = f"{self.names[int(cls)]} {confidence:.2f}"
                            image = cv2.putText(image, label, [pt1[0] + 10, pt1[1] + 10], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        if int(cls) == 7:
                            box_pix = (np.array(pt1) + np.array(pt2)) / 2
        if not box_pix is None and not self.debuger is None:
            self.debuger.update_param("box_R", np.linalg.norm(self.pc.pix2base(box_pix, "box")[:2, 0] - self.pc.pix2base(self.camera_centre, "box")[:2, 0]))
        # 绘制自动识别目标的像素位置
        if not ball_pix is None:
            image = cv2.circle(image, ball_pix, 3, (0, 0, 255), -1)
            image = cv2.putText(image, "TB", ball_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if not self.debuger is None:
                self.debuger.update_param("ball_R", np.linalg.norm(self.pc.pix2base(ball_pix, "ball")[:2, 0] - self.pc.pix2base(self.camera_centre, "ball")[:2, 0]))
        if not cube_pix is None:
            image = cv2.circle(image, cube_pix, 3, (0, 0, 255), 2)
            image = cv2.putText(image, "TC", cube_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if not self.debuger is None:
                self.debuger.update_param("cube_R", np.linalg.norm(self.pc.pix2base(cube_pix, "cube")[:2, 0] - self.pc.pix2base(self.camera_centre, "cube")[:2, 0]))
        # 绘制手动输入目标的像素位置
        if not self.click_point is None:
            image = cv2.circle(image, self.click_point, 3, (255, 255, 255), -1)
            image = cv2.circle(image, self.click_point, 2, (255, 0, 0), -1)
            image = cv2.putText(image, "manual", self.click_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if not self.debuger is None:
                self.debuger.update_param("manual_R", np.linalg.norm(self.pc.pix2base(self.click_point, "floor") - self.pc.pix2base(self.camera_centre, "floor")))
        # 绘制双击输入目标的像素位置
        if not self.double_click_point is None:
            image = cv2.circle(image, self.double_click_point, 3, (0, 255, 0), -1)
            image = cv2.putText(image, "place", self.double_click_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 绘制相机中心
        image = cv2.circle(image, self.camera_centre, 1, (150, 0, 255), -1)
        # 绘制世界系原点在地面上的投影点在像素系下的位置
        image = cv2.circle(image, self.origin_pix, 1, (255, 0, 0), -1)
        # 绘制抓取范围
        image[self.ball_pts[:, 1], self.ball_pts[:, 0]] = [0, 0, 255]
        image[self.box_pts[:, 1], self.box_pts[:, 0]] = [255, 0, 0]
        # 绘制准星
        aimed_centre = self.pc.base2pix(self.pc.polar2base([self.R, -np.pi / 2], "floor"))
        left = np.array(aimed_centre - np.array([5, 0]), dtype=np.uint16)
        right = np.array(aimed_centre + np.array([5, 0]), dtype=np.uint16)
        up = np.array(aimed_centre - np.array([0, 5]), dtype=np.uint16)
        down = np.array(aimed_centre + np.array([0, 5]), dtype=np.uint16)
        image = cv2.line(image, left, right, (255, 0, 0), 2)
        image = cv2.line(image, up, down, (255, 0, 0), 2)

        # 修复1：OpenCV的resize函数要求尺寸为(width, height)，而非(height, width)
        # 计算新尺寸时交换宽高顺序
        new_width = int(self.current_reshape_ratio * image.shape[1])  # 宽度
        new_height = int(self.current_reshape_ratio * image.shape[0]) # 高度
        new_shape = (new_width, new_height)  # 正确的尺寸格式(width, height)

        # 修复2：使用INTER_LINEAR插值确保放大效果更清晰
        image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)

        # 修复3：确保QLabel能显示完整图像，不限制尺寸
        now_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = now_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(now_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
    
        # 设置图像自适应QLabel大小，同时保持比例
        self.image_label.setScaledContents(False)  # 关闭自动缩放
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        ))

    def update_log(self, log):
        cursor = self.log_widget.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(log)
        
        # 滚动到底部
        self.log_widget.ensureCursorVisible()

    def update_rec(self, rec_data):
        cmd, theta, self.R = rec_data
        self.status.setText(f"电控当前状态：{cmd}\nR：{self.R}\ntheta：{theta}")

    def update_icon(self, icon):
        height, width = [100, 100]
        bytes_per_line = 3 * width
        q_image = QImage(icon[0], width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.icon_image.setPixmap(pixmap)

    def update_box_color(self, detect_res):
        target_list = []
        for target in detect_res:
            if 7 <= target[-1] <= 12:
                target_list.append({"class":target[-1], "confidence":target[-2]})
        if target_list:
            target_list = sorted(target_list, key= lambda x:x["confidence"], reverse=True)
            match target_list[0]["class"]:
                case 7:
                    now_color = "yellow"
                case 8:
                    now_color = "green"
                case 9:
                    now_color = "brown"
                case 10:
                    now_color = "blue"
                case 11:
                    now_color = "pink"
                case 12:
                    now_color = "black"
                case _:
                    now_color = "white"
            if not self.box_color_queue or (now_color != "white" and now_color != self.box_color_queue[-1]):
                if self.auto_record_color:
                    self.click_color = now_color
                    self.detect_color_show.setStyleSheet(f"background-color: {now_color};")
                self.box_color_queue.append(now_color)
                # if len(self.box_color_queue) > 10:
                #     del self.box_color_queue[0]
                box_color_text = ""
                for color in self.box_color_queue:
                    match color:
                        case "yellow":
                            box_color_text += "黄 "
                        case "green":
                            box_color_text += "绿 "
                        case "brown":
                            box_color_text += "棕 "
                        case "blue":
                            box_color_text += "蓝 "
                        case "pink":
                            box_color_text += "粉 "
                        case "black":
                            box_color_text += "黄 "
                        case _:
                            continue
                self.color_his_text.setPlainText(box_color_text)

    # 同时修改鼠标点击坐标计算（保持比例一致性）
    def on_label_click(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if pixmap:
                # 获取标签和图像的实际尺寸
                label_size = self.image_label.size()
                # 获取图片的实际大小
                pixmap_size = pixmap.size()

                # 计算缩放比例（图像实际显示尺寸 / 原始图像尺寸）
                scale_width = pixmap_size.width() / self.image_size[1]
                scale_height = pixmap_size.height() / self.image_size[0]

                # 计算在原始图像上的坐标（修正比例计算）
                image_pos = [
                    int((pos.x() * self.image_size[1]) / pixmap_size.width()),
                    int((pos.y() * self.image_size[0]) / pixmap_size.height())
                ]
            
                if 0 <= image_pos[0] < self.image_size[1] and 0 <= image_pos[1] < self.image_size[0]:
                    if event.button() == Qt.LeftButton:
                        self.publisher_target.publish(image_pos)
                        self.click_point = image_pos
                    else:
                        self.publisher_dou_tar.publish(image_pos)
                        self.double_click_point = image_pos

    def keyPressEvent(self, event):
        # 添加F11快捷键用于切换全屏
        if event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()  # 退出全屏
            else:
                self.showFullScreen()  # 进入全屏
            return
            
        if event.isAutoRepeat():
            return
        # 处理回车键（切换图像归一化）
        if event.key() in [Qt.Key_Enter, Qt.Key_Return]:
            self.normalization = not self.normalization
        # A键手动抓球
        elif event.key() == Qt.Key_A:
            self.catch_ball_callback()
        # D键手动抓魔方
        elif event.key() == Qt.Key_D:
            self.catch_cube_callback()
        # W放下魔方
        elif event.key() == Qt.Key_W:
            self.throw_cube_callback()
        # S放下球
        elif event.key() == Qt.Key_S:
            self.throw_ball_callback()

    # 窗口大小变化事件，用于检测全屏状态
    def resizeEvent(self, event):
        # 根据窗口是否全屏更新缩放比例
        if self.isFullScreen():
            self.current_reshape_ratio = 2.0  # 全屏时放大到2倍
        else:
            self.current_reshape_ratio = self.reshape_ratio  # 恢复配置比例
        super().resizeEvent(event)

    def AI_catch_ball_callback(self): # 自动抓球
        self.publisher_action.publish(1) 
    
    def AI_throw_ball_callback(self):
        self.publisher_action.publish(3) # 自动放球

    def AI_catch_cube_callback(self):
        self.publisher_action.publish(2) # 自动抓魔方
    
    def AI_throw_cube_callback(self): # 自动放魔方
        self.publisher_action.publish(4)
    
    def move_cube_callback(self): # 自动扫堂腿
        self.publisher_action.publish(5)

    def loose_callback(self): # 自动复位
        self.publisher_action.publish(6)

    def AI_edge_ball_callback(self): # 自动边沿抓球
        self.publisher_action.publish(7)

    def catch_ball_callback(self): # 手动抓球
        self.publisher_action.publish(9) 
    
    def throw_ball_callback(self):
        self.publisher_action.publish(11) # 手动放球

    def catch_cube_callback(self):
        self.publisher_action.publish(10) # 手动抓魔方
    
    def throw_cube_callback(self): # 手动放魔方
        self.publisher_action.publish(12)
    
    def raise_arm_callback(self): # 上提
        self.publisher_action.publish(13)

    def reset_callback(self): # 电控复位
        self.publisher_action.publish(14)

    def edge_ball_callback(self): # 手动边沿抓球
        self.publisher_action.publish(15)

    def restart_camera_callback(self): # 重启相机
        self.restart_camera.emit()
    
    def counterclockwise_90_callback(self): # 逆时针旋转90度
        self.publisher_action.publish(-1)

    def turn_180_callback(self): # 旋转180度
        self.publisher_action.publish(-2)

    def clockwise_90_callback(self): # 顺时针旋转90度
        self.publisher_action.publish(-3)

    def return_origin_callback(self): # 顺时针旋转90度
        self.publisher_action.publish(-6)
    
    def go_end_callback(self):
        self.publisher_action.publish(-7)

    def clear_callback(self): # 清空日志
        self.log_widget.clear()

    def stop_task_callback(self): # 停止任务
        self.publisher_action.publish(-4)
    
    def clear_singlechip_callback(self):
        self.publisher_action.publish(-5)

    def clear_color_callback(self):
        self.color_his = ["white"] * 6
        np.savetxt(self.color_his_path, np.array(self.color_his), fmt='%s')
        self.ball_box_button_1.setStyleSheet("color: black; background-color: white;")
        self.ball_box_button_2.setStyleSheet("color: black; background-color: white;")
        self.ball_box_button_3.setStyleSheet("color: black; background-color: white;")
        self.ball_box_button_4.setStyleSheet("color: black; background-color: white;")
        self.ball_box_button_5.setStyleSheet("color: black; background-color: white;")
        self.ball_box_button_6.setStyleSheet("color: black; background-color: white;")
    def set_yellow(self):
        if not self.auto_record_color:
            self.click_color = "yellow"
            self.detect_color_show.setStyleSheet(f"background-color: yellow;")

    def set_brown(self):
        if not self.auto_record_color:
            self.click_color = "brown"
            self.detect_color_show.setStyleSheet(f"background-color: brown;")

    def set_green(self):
        if not self.auto_record_color:
            self.click_color = "green"
            self.detect_color_show.setStyleSheet(f"background-color: green;")
    
    def set_black(self):
        if not self.auto_record_color:
            self.click_color = "black"
            self.detect_color_show.setStyleSheet(f"background-color: black;")
    
    def set_pink(self):
        if not self.auto_record_color:
            self.click_color = "pink"
            self.detect_color_show.setStyleSheet(f"background-color: pink;")

    def set_blue(self):
        if not self.auto_record_color:
            self.click_color = "blue"
            self.detect_color_show.setStyleSheet(f"background-color: blue;")

    def change_record_color_status_callback(self):
        if self.auto_record_color:
            self.auto_record_button.setText('记录颜色：手动')
            self.auto_record_button.setStyleSheet('color: black;')
            self.auto_record_color = False
        else:
            self.auto_record_button.setText('记录颜色：自动')
            self.auto_record_button.setStyleSheet('color: red;')
            self.auto_record_color = True

    def update_color(self, data):
        self.color_his[int(data[0]) - 1] = data[1]
        np.savetxt(self.color_his_path, np.array(self.color_his), fmt='%s')

    def get_color(self, type:int):
        match type:
            case 0 | 13:
                return (255, 255, 255)
            case 1 | 7:
                return (0, 255, 255)
            case 2 | 8:
                return (0, 128, 0)
            case 3 | 9:
                return (0, 74, 186)
            case 4 | 10:
                return (255, 0, 0)
            case 5 | 11:
                return (252, 163, 255)
            case 6 | 12:
                return (0, 0, 0)
            case _:
                return (255, 255, 255)

# 定义颜色按钮
class ColorButton(QPushButton):
    update_color_signal = pyqtSignal(list)
    def __init__(self, parent:Main_window, name=None):
        super().__init__(name)
        self.my_parent = parent
        self.name = name
        self.color = "white"
        
    def set_selected_color(self):
        if self.my_parent.auto_record_color and self.my_parent.box_color_queue:
            self.color = self.my_parent.box_color_queue[-1]
        elif not self.my_parent.auto_record_color:
            self.color = self.my_parent.click_color
        if self.color == "black":
            self.setStyleSheet(f"color: white; background-color: {self.color};")
        else:
            self.setStyleSheet(f"color: black; background-color: {self.color};")  # 设置按钮颜色为指定的颜色
        self.update_color_signal.emit([self.name, self.color])