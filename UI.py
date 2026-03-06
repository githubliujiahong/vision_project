"""
@Module: UI
@Description: 视觉系统的领航员交互界面。
              基于 PyQt5 构建，负责多线程拉取视觉处理结果并进行图形化渲染，
              同时提供指令下发面板，供操作员进行半自动/手动干预控制。
"""

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QMainWindow, QLabel, QPushButton, QGroupBox, QTextEdit
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2, toml, Bridge, os
from post_process import Position_calculate
import numpy as np


class UI_thread(QThread):
    """
    UI 数据更新线程。
    后台高频拉取各个 Bridge 话题的数据，并通过 PyQt 信号机制安全地发送给主 GUI 线程。
    """
    update_image_signal = pyqtSignal(list)
    update_log_signal = pyqtSignal(str)
    update_rec_signal = pyqtSignal(list)
    update_icon_signal = pyqtSignal(list)
    update_color_signal = pyqtSignal(np.ndarray)

    def __init__(self, show_icon):
        super().__init__()
        self.subscriber = Bridge.Subscriber("image", 1)
        self.detect_sub = Bridge.Subscriber("detect_res", 1)
        self.ball_sub = Bridge.Subscriber("ball_pix", 1)
        self.cube_sub = Bridge.Subscriber("cube_pix", 1)
        self.log_sub = Bridge.Subscriber("log", 10)
        self.rec_sub = Bridge.Subscriber("rec_data", 1)
        if show_icon:
            self.icon_sub = Bridge.Subscriber("icon", 1)

        self.stop = False
        self.detect_res = None
        self.ball_pix = None
        self.cube_pix = None
        self.show_icon = show_icon

    def run(self):
        while not self.stop:
            # 1. 抓取主画面 (严格超时控制)
            try:
                now_image = self.subscriber.get_message(0.05)
            except Exception:
                continue

            # 2. 抓取各类附属数据
            try:
                self.detect_res = self.detect_sub.get_message(0.003)
                self.update_color_signal.emit(self.detect_res)
            except Exception:
                pass

            try:
                self.ball_pix = self.ball_sub.get_message(0.003)
                self.ball_pix = [int(self.ball_pix[0]), int(self.ball_pix[1])]
            except Exception:
                pass

            try:
                self.cube_pix = self.cube_sub.get_message(0.003)
                self.cube_pix = [int(self.cube_pix[0]), int(self.cube_pix[1])]
            except Exception:
                pass

            # 3. 抓取日志并拼接
            try:
                log = ""
                while not self.log_sub.is_empty():
                    log += self.log_sub.get_message(0.003) + '\n'
                if log:
                    self.update_log_signal.emit(log)
            except Exception:
                pass

            try:
                rec_data = self.rec_sub.get_message(0.003)
                self.update_rec_signal.emit(rec_data)
            except Exception:
                pass

            if self.show_icon:
                try:
                    icon = self.icon_sub.get_message(0.003)
                    self.update_icon_signal.emit([icon])
                except Exception:
                    pass

            # 统一打包发送渲染信号
            self.update_image_signal.emit([now_image, self.cube_pix, self.ball_pix, self.detect_res])

    def end(self):
        self.stop = True


class Main_window(QMainWindow):
    restart_camera = pyqtSignal()

    def __init__(self, post_process_th, debuger):
        super().__init__()
        self.setWindowTitle("MAN!!! What Can I Say?")

        # ==========================================
        # 1. 配置加载与数学工具初始化
        # ==========================================
        ROOT = os.getcwd()
        self.config = toml.load(os.path.join(ROOT, "config/UI.toml"))
        self.show_icon = self.config["show_icon"]
        self.color_his_path = self.config["color_his_path"]
        self.reshape_ratio = self.config["reshape_ratio"]
        self.current_reshape_ratio = self.reshape_ratio

        config = toml.load(os.path.join(ROOT, "config/config.toml"))
        specific_config = config["specific_config"]
        PATH = os.path.join(ROOT, "config", specific_config).replace('/', os.sep).replace("\\", os.sep)
        sys_config = toml.load(PATH)

        # 相机解算参数准备
        K = np.array(sys_config["camera"]["K"], dtype=np.float32)
        D = np.array(sys_config["camera"]["D"], dtype=np.float32)
        alpha = sys_config["camera"]["alpha"]
        self.image_size = sys_config["camera"]["image_shape"]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, self.image_size, alpha, self.image_size, True)
        self.camera_centre = np.array([int(new_K[0, 2]), int(new_K[1, 2])])
        self.arm_length = sys_config["decision"]["arm_length"]
        self.reach = 30

        self.post_process_th = post_process_th
        self.debuger = debuger
        if self.debuger is not None:
            self.debuger.add_param("manual_R", 1)
            self.debuger.add_param("ball_R", 10)
            self.debuger.add_param("cube_R", 10)
            self.debuger.add_param("box_R", 10)

        # 核心交互状态参数
        self.use_AI = True
        self.click_point = None
        self.double_click_point = None
        self.normalization = False
        self.box_color_queue = []
        self.box_color_text = ""
        self.auto_record_color = False
        self.R = 0

        # 加载颜色历史记录
        try:
            tem = np.loadtxt(self.color_his_path, dtype=str)
            self.color_his = tem.tolist()
            valid_colors = {"white", "black", "pink", "brown", "blue", "yellow", "green"}
            for color in self.color_his:
                if color not in valid_colors:
                    raise Exception(f"unexpected value: {color}")
            print("[UI] Find color history successfully.")
        except Exception as err:
            print(f"[UI] Color history load failed: {err}")
            self.color_his = ["white"] * 6

        # 坐标计算器实例
        self.pc = Position_calculate.Position_calculator(None)
        self.origin_pix = np.array(self.pc.get_base_origin_in_pix("floor"), dtype=np.uint)
        self.ball_pts = self.pc.get_rotation_elliptic_pts("ball", self.image_size)
        self.box_pts = self.pc.get_rotation_elliptic_pts("box", self.image_size)

        # ==========================================
        # 2. UI 控件构建 (模块化与 DRY 原则重构)
        # ==========================================
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout()

        # --- 2.1 图像显示区 ---
        image_layout = QVBoxLayout()
        tool_layout = QHBoxLayout()

        if self.show_icon:
            self.icon_image = QLabel()
            q_image = QImage(np.zeros((100, 100, 3), dtype=np.uint8), 100, 100, 300, QImage.Format_RGB888)
            self.icon_image.setPixmap(QPixmap.fromImage(q_image))
            tool_layout.addWidget(self.icon_image)

        self.restart_buttom = QPushButton("重启相机")
        self.restart_buttom.clicked.connect(self.restart_camera_callback)
        tool_layout.addWidget(self.restart_buttom)

        self.image_label = QLabel()
        height, width = self.image_size
        q_image = QImage(np.zeros((height, width, 3), dtype=np.uint8), width, height, 3 * width, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_image))
        self.image_label.mousePressEvent = self.on_label_click

        image_layout.addLayout(tool_layout)
        image_layout.addWidget(self.image_label)
        main_layout.addLayout(image_layout)

        # --- 2.2 工作控制区 ---
        workspace_layout = QVBoxLayout()

        # (1) 颜色状态记录模块
        color_his_layout = QHBoxLayout()
        self.detect_color_show = QPushButton()
        self.detect_color_show.setStyleSheet("background-color: white;")
        self.detect_color_show.setFixedSize(100, 100)

        color_his_right_layout = QVBoxLayout()
        self.auto_record_button = QPushButton('记录颜色：手动')
        self.auto_record_button.clicked.connect(self.change_record_color_status_callback)

        self.color_his_text = QTextEdit()
        self.color_his_text.setReadOnly(True)

        color_his_right_layout.addWidget(self.auto_record_button)
        color_his_right_layout.addWidget(self.color_his_text)
        color_his_layout.addWidget(self.detect_color_show)
        color_his_layout.addLayout(color_his_right_layout)

        # (2) 目标颜色选择模块 (DRY 优化: 循环生成)
        color_select_layout = QHBoxLayout()
        ui_colors = ["yellow", "blue", "pink", "black", "green", "brown"]
        for color_name in ui_colors:
            btn = QPushButton()
            btn.setStyleSheet(f"background-color: {color_name}")
            btn.setFixedSize(50, 100)
            # 使用闭包绑定当前颜色参数
            btn.clicked.connect(self._create_color_setter(color_name))
            color_select_layout.addWidget(btn)

        # (3) 小球排列记录模块 (DRY 优化: 循环生成)
        self.click_color = "white"
        color_group_box = QGroupBox("小球颜色 (记录)")
        color_layout_inner = QVBoxLayout()

        self.clear_color_button = QPushButton("清空颜色记录")
        self.clear_color_button.setStyleSheet('color: red;')
        self.clear_color_button.clicked.connect(self.clear_color_callback)
        color_layout_inner.addWidget(self.clear_color_button)

        ball_first_layout = QHBoxLayout()
        ball_second_layout = QHBoxLayout()
        self.ball_box_buttons = []

        for i in range(1, 7):
            btn = ColorButton(self, str(i))
            btn.clicked.connect(btn.set_selected_color)
            bg_color = self.color_his[i - 1]
            text_color = "white" if bg_color == "black" else "black"
            btn.setStyleSheet(f"color: {text_color}; background-color: {bg_color}")
            btn.update_color_signal.connect(self.update_color)
            self.ball_box_buttons.append(btn)

            if i <= 3:
                ball_first_layout.addWidget(btn)
            else:
                ball_second_layout.addWidget(btn)

        color_layout_inner.addLayout(ball_first_layout)
        color_layout_inner.addLayout(ball_second_layout)
        color_group_box.setLayout(color_layout_inner)

        # (4) 机器狗行为控制面板 (AI/手动/花活)
        workspace_layout.addLayout(self._build_action_panel("自动模式", [
            ("抓取球", self.AI_catch_ball_callback), ("放下球", self.AI_throw_ball_callback),
            ("抓取魔方", self.AI_catch_cube_callback), ("放下魔方", self.AI_throw_cube_callback),
            ("抓边沿球", self.AI_edge_ball_callback)
        ]))
        workspace_layout.addLayout(self._build_action_panel("手动模式", [
            ("抓取球", self.catch_ball_callback), ("放下球", self.throw_ball_callback),
            ("抓取魔方", self.catch_cube_callback), ("放下魔方", self.throw_cube_callback),
            ("抓边沿球", self.edge_ball_callback)
        ]))
        workspace_layout.addLayout(self._build_action_panel("高级动作(花活)", [
            ("原地松爪", self.loose_callback), ("四两拨千斤", self.move_cube_callback),
            ("舵臂上提", self.raise_arm_callback)
        ]))

        workspace_layout.addLayout(color_his_layout)
        workspace_layout.addWidget(color_group_box)
        workspace_layout.addLayout(color_select_layout)
        main_layout.addLayout(workspace_layout)

        # --- 2.3 状态监控区 ---
        status_layout = QVBoxLayout()

        status_layout.addLayout(self._build_action_panel("底盘位置调整", [
            ("逆时针旋转90°", self.counterclockwise_90_callback),
            ("顺时针旋转90°", self.clockwise_90_callback),
            ("旋转180°", self.turn_180_callback), ("回到原点", self.return_origin_callback),
            ("移开抓", self.go_end_callback), ("电控复位", self.reset_callback)
        ], horizontal_first=True))

        self.status = QLabel("电控当前状态：null\nR：null\ntheta：null")
        status_layout.addWidget(self.status)
        status_layout.addWidget(QLabel("系统日志"))

        btn_clear = QPushButton("清空日志");
        btn_clear.clicked.connect(self.log_widget_clear)
        btn_stop = QPushButton("停止任务");
        btn_stop.clicked.connect(self.stop_task_callback)
        btn_clear_mcu = QPushButton("清空单片机指令");
        btn_clear_mcu.clicked.connect(self.clear_singlechip_callback)

        status_layout.addWidget(btn_clear)
        status_layout.addWidget(btn_stop)
        status_layout.addWidget(btn_clear_mcu)

        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumWidth(300)
        status_layout.addWidget(self.log_widget)

        main_layout.addLayout(status_layout)

        # 布局权重设置
        main_layout.setStretch(0, 3)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 1)
        self.central_widget.setLayout(main_layout)

        # ==========================================
        # 3. 通信与线程装配
        # ==========================================
        self.publisher_target = Bridge.Publisher("manual_target")
        self.publisher_dou_tar = Bridge.Publisher("double_target")
        self.publisher_action = Bridge.Publisher("action")

        self.ui_thread = UI_thread(self.show_icon)
        self.ui_thread.update_image_signal.connect(self.update_image)
        self.ui_thread.update_log_signal.connect(self.update_log)
        self.ui_thread.update_rec_signal.connect(self.update_rec)
        self.ui_thread.update_color_signal.connect(self.update_box_color)
        if self.show_icon:
            self.ui_thread.update_icon_signal.connect(self.update_icon)

    # ==========================================
    # UI 构建辅助函数 (DRY 优化)
    # ==========================================
    def _create_color_setter(self, color_name):
        """闭包工厂：生成对应颜色的点击回调函数"""

        def setter():
            if not self.auto_record_color:
                self.click_color = color_name
                self.detect_color_show.setStyleSheet(f"background-color: {color_name};")

        return setter

    def _build_action_panel(self, title, actions, horizontal_first=False):
        """快速生成控制面板 GroupBox 的辅助工厂"""
        layout = QHBoxLayout()
        group_box = QGroupBox(title)
        inner_layout = QVBoxLayout()

        current_sub_layout = None
        for i, (text, callback) in enumerate(actions):
            btn = QPushButton(text)
            btn.clicked.connect(callback)

            # 简单的两两并排逻辑
            if not horizontal_first:
                if i % 2 == 0:
                    current_sub_layout = QHBoxLayout()
                    inner_layout.addLayout(current_sub_layout)
                current_sub_layout.addWidget(btn)
            else:
                inner_layout.addWidget(btn)

        group_box.setLayout(inner_layout)
        layout.addWidget(group_box)
        return layout

    # ==========================================
    # 核心渲染与状态更新
    # ==========================================
    def update_image(self, bag):
        image, cube_pix, ball_pix, detect_res = bag
        if image is None or image.size == 0:
            return

        aimed_centre = self.pc.base2pix(self.pc.polar2base([self.R, -np.pi / 2], "floor"))

        # --- 绘制 YOLO 识别框 ---
        if detect_res is not None and len(detect_res) > 0:
            try:
                for det in detect_res:
                    if len(det) < 6: continue
                    x1, y1, x2, y2 = map(int, det[:4])
                    conf, cls_id = float(det[4]), int(det[5])
                    h, w = image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, f"ID:{cls_id} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 0, 255), 1)
            except Exception as e:
                print(f"[UI Draw Error] {e}")

        # --- 绘制落点准星与标线 ---
        if self.click_point is not None:
            image = cv2.circle(image, self.click_point, 1, (255, 0, 0), -1)
            image = cv2.putText(image, "manual", self.click_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if self.debuger is not None:
                self.debuger.update_param("manual_R", np.linalg.norm(
                    self.pc.pix2base(self.click_point, "floor") - self.pc.pix2base(self.camera_centre, "floor")))

            ax, ay = self.click_point
            bx, by = aimed_centre
            dx, dy = bx - ax, by - ay

            if dx == 0 and dy == 0:
                cv2.line(image, (int(ax - self.reach - 5), ay), (int(ax - self.reach + 5), ay), (0, 0, 255), 2)
                cv2.line(image, (ax, int(ay - self.reach - 5)), (ax, int(ay - self.reach + 5)), (0, 0, 255), 2)
            else:
                norm = np.sqrt(dx ** 2 + dy ** 2)
                ux, uy = dx / norm, dy / norm
                perp_ux, perp_uy = -uy, ux
                # 计算四个线条的中心点
                centers = [
                    (int(ax + ux * self.reach), int(ay + uy * self.reach)),
                    (int(ax - ux * self.reach), int(ay - uy * self.reach)),
                    (int(ax + perp_ux * self.reach), int(ay + perp_uy * self.reach)),
                    (int(ax - perp_ux * self.reach), int(ay - perp_uy * self.reach))
                ]
                # 绘制辅助捕捉线
                for cx, cy in centers[:2]:
                    cv2.line(image, (int(cx - perp_ux * 5), int(cy - perp_uy * 5)),
                             (int(cx + perp_ux * 5), int(cy + perp_uy * 5)), (0, 0, 255), 2)
                for cx, cy in centers[2:]:
                    cv2.line(image, (int(cx - ux * 5), int(cy - uy * 5)), (int(cx + ux * 5), int(cy + uy * 5)),
                             (0, 0, 255), 2)

        if self.double_click_point is not None:
            image = cv2.circle(image, self.double_click_point, 1, (0, 255, 0), -1)
            image = cv2.putText(image, "place", self.double_click_point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                1)

        image = cv2.circle(image, self.camera_centre, 1, (150, 0, 255), -1)
        image = cv2.circle(image, self.origin_pix, 1, (255, 0, 0), -1)
        image[self.ball_pts[:, 1], self.ball_pts[:, 0]] = [0, 0, 255]
        image[self.box_pts[:, 1], self.box_pts[:, 0]] = [255, 0, 0]

        # 图像缩放与呈现
        new_shape = (int(self.current_reshape_ratio * image.shape[1]), int(self.current_reshape_ratio * image.shape[0]))
        image = cv2.resize(image, new_shape, interpolation=cv2.INTER_LINEAR)
        now_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        q_image = QImage(now_image.data, now_image.shape[1], now_image.shape[0], 3 * now_image.shape[1],
                         QImage.Format_RGB888)
        self.image_label.setScaledContents(False)
        self.image_label.setPixmap(
            QPixmap.fromImage(q_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_log(self, log):
        cursor = self.log_widget.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(log)
        self.log_widget.ensureCursorVisible()

    def update_rec(self, rec_data):
        cmd, theta, self.R = rec_data
        self.status.setText(f"电控当前状态：{cmd}\nR：{self.R}\ntheta：{theta}")

    def update_icon(self, icon):
        q_image = QImage(icon[0], 100, 100, 300, QImage.Format_RGB888)
        self.icon_image.setPixmap(QPixmap.fromImage(q_image))

    def update_box_color(self, detect_res):
        """处理检测结果中的球体颜色序列"""
        target_list = [{"class": t[-1], "confidence": t[-2]} for t in detect_res if 7 <= t[-1] <= 12]
        if not target_list:
            return

        target_list.sort(key=lambda x: x["confidence"], reverse=True)
        current_class = target_list[0]["class"]

        # DRY: 使用字典映射替代一长串 if-elif
        color_map = {7: "yellow", 8: "green", 9: "brown", 10: "blue", 11: "pink", 12: "black"}
        cn_color_map = {"yellow": "黄", "green": "绿", "brown": "棕", "blue": "蓝", "pink": "粉", "black": "黑"}

        now_color = color_map.get(current_class, "white")

        if not self.box_color_queue or (now_color != "white" and now_color != self.box_color_queue[-1]):
            if self.auto_record_color:
                self.click_color = now_color
                self.detect_color_show.setStyleSheet(f"background-color: {now_color};")
            self.box_color_queue.append(now_color)

            box_color_text = " ".join([cn_color_map.get(c, "") for c in self.box_color_queue if c in cn_color_map])
            self.color_his_text.setPlainText(box_color_text)

    def on_label_click(self, event):
        if event.button() in [Qt.LeftButton, Qt.RightButton]:
            pos = event.pos()
            pixmap = self.image_label.pixmap()
            if not pixmap: return

            label_size, pixmap_size = self.image_label.size(), pixmap.size()
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
        if event.key() == Qt.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
            return
        if event.isAutoRepeat(): return

        key_map = {
            Qt.Key_Enter: lambda: setattr(self, 'normalization', not self.normalization),
            Qt.Key_Return: lambda: setattr(self, 'normalization', not self.normalization),
            Qt.Key_A: self.catch_ball_callback,
            Qt.Key_D: self.catch_cube_callback,
            Qt.Key_W: self.throw_cube_callback,
            Qt.Key_S: self.throw_ball_callback,
            Qt.Key_F: self.edge_ball_callback
        }
        if event.key() in key_map:
            key_map[event.key()]()

    def resizeEvent(self, event):
        self.current_reshape_ratio = 2.0 if self.isFullScreen() else self.reshape_ratio
        super().resizeEvent(event)

    # ==========================================
    # 动作发送回调组 (与底层协议对接)
    # ==========================================
    def AI_catch_ball_callback(self):
        self.publisher_action.publish(1)

    def AI_catch_cube_callback(self):
        self.publisher_action.publish(2)

    def AI_throw_ball_callback(self):
        self.publisher_action.publish(3)

    def AI_throw_cube_callback(self):
        self.publisher_action.publish(4)

    def move_cube_callback(self):
        self.publisher_action.publish(5)

    def loose_callback(self):
        self.publisher_action.publish(6)

    def AI_edge_ball_callback(self):
        self.publisher_action.publish(7)

    def catch_ball_callback(self):
        self.publisher_action.publish(9)

    def catch_cube_callback(self):
        self.publisher_action.publish(10)

    def throw_ball_callback(self):
        self.publisher_action.publish(11)

    def throw_cube_callback(self):
        self.publisher_action.publish(12)

    def raise_arm_callback(self):
        self.publisher_action.publish(13)

    def reset_callback(self):
        self.publisher_action.publish(14)

    def edge_ball_callback(self):
        self.publisher_action.publish(15)

    def counterclockwise_90_callback(self):
        self.publisher_action.publish(-1)

    def turn_180_callback(self):
        self.publisher_action.publish(-2)

    def clockwise_90_callback(self):
        self.publisher_action.publish(-3)

    def stop_task_callback(self):
        self.publisher_action.publish(-4)

    def clear_singlechip_callback(self):
        self.publisher_action.publish(-5)

    def return_origin_callback(self):
        self.publisher_action.publish(-6)

    def go_end_callback(self):
        self.publisher_action.publish(-7)

    def restart_camera_callback(self):
        self.restart_camera.emit()

    def log_widget_clear(self):
        self.log_widget.clear()

    def clear_color_callback(self):
        self.color_his = ["white"] * 6
        np.savetxt(self.color_his_path, np.array(self.color_his), fmt='%s')
        for btn in getattr(self, "ball_box_buttons", []):
            btn.setStyleSheet("color: black; background-color: white;")

    def change_record_color_status_callback(self):
        self.auto_record_color = not self.auto_record_color
        if self.auto_record_color:
            self.auto_record_button.setText('记录颜色：自动')
            self.auto_record_button.setStyleSheet('color: red;')
        else:
            self.auto_record_button.setText('记录颜色：手动')
            self.auto_record_button.setStyleSheet('color: black;')

    def update_color(self, data):
        self.color_his[int(data[0]) - 1] = data[1]
        np.savetxt(self.color_his_path, np.array(self.color_his), fmt='%s')

    def start(self):
        self.ui_thread.start()

    def end(self):
        self.ui_thread.end()

    def wait(self):
        self.ui_thread.wait()


class ColorButton(QPushButton):
    update_color_signal = pyqtSignal(list)

    def __init__(self, parent: Main_window, name=None):
        super().__init__(name)
        self.my_parent = parent
        self.name = name
        self.color = "white"

    def set_selected_color(self):
        if self.my_parent.auto_record_color and self.my_parent.box_color_queue:
            self.color = self.my_parent.box_color_queue[-1]
        elif not self.my_parent.auto_record_color:
            self.color = self.my_parent.click_color

        text_color = "white" if self.color == "black" else "black"
        self.setStyleSheet(f"color: {text_color}; background-color: {self.color};")
        self.update_color_signal.emit([self.name, self.color])