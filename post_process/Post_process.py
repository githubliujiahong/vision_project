"""
@Module: Post_process
@Description: 视觉系统的决策与空间解算中枢 (Decision Maker & Position Calculator)。
              基于 YOLO 的像素级 2D 检测结果，结合相机内参和坐标系转换矩阵，
              解算出目标在现实世界中的 3D 极坐标。
              包含：目标评分筛选、多条件稳定排序、2D 向量势场避障等核心算法。
"""

import os
import sys
import numpy as np
import toml
import time
from PyQt5.QtCore import QThread

PKG_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PKG_PATH)

import Bridge
import Position_calculate


class Mean_filter():
    """滑动窗口平均滤波器，用于平滑坐标数据（当前决策流暂时备用）"""

    def __init__(self, max_size=np.inf) -> None:
        self.__max_size = max_size
        self.__data = []

    def push(self, data) -> None:
        while len(self.__data) >= self.__max_size:
            del self.__data[0]
        self.__data.append(data)

    def get_average(self) -> float:
        if not self.__data:
            return 0.0
        return sum(self.__data) / len(self.__data)


class Post_processor(QThread):
    def __init__(self, debuger=None) -> None:
        super().__init__()

        # --- 1. 级联配置加载 ---
        ROOT = os.getcwd()
        config = toml.load(os.path.join(ROOT, "config/config.toml"))
        PATH = os.path.join(ROOT, "config", config["specific_config"]).replace('/', os.sep).replace("\\", os.sep)
        self.config = toml.load(PATH)

        # --- 2. 坐标解算器初始化 (注入相机畸变内参) ---
        position_config = self.config["position_calculate"]
        position_config['K'] = self.config["camera"]['K']
        position_config['D'] = self.config["camera"]['D']
        self.position_calculator = Position_calculate.Position_calculator(debuger)

        # --- 3. 线程控制与通信总线建立 ---
        self.stop = False
        self.debuger = debuger

        # 订阅流 (获取前端数据)
        self.subscriber_detect = Bridge.Subscriber("detect_res")
        self.subscriber_manual = Bridge.Subscriber("manual_target")
        # 下面三个话题如果后续不扩展，可作为保留接口
        self.subscriber_place = Bridge.Subscriber("double_target")
        self.subscriber_action = Bridge.Subscriber("action")
        self.subscriber_rec = Bridge.Subscriber("rec_data")

        # 发布流 (向下位机/UI 发送数据)
        self.publisher = Bridge.Publisher("send_data")
        self.publisher_log = Bridge.Publisher("log")

    def run(self) -> None:
        """决策线程主循环"""
        img_h, img_w = self.config["camera"]["image_shape"]
        center_x, center_y = img_w / 2, img_h / 2

        while not self.stop:
            # ---------------------------------------------------------
            # 第一阶段：非阻塞式信号触发 (消除 TOCTOU 漏洞)
            # ---------------------------------------------------------
            triggered = False

            # 1. 监听手动触发信号 (EAFP 模式，直接拿)
            try:
                msg = self.subscriber_manual.get_message(timeout=0.001)
                if msg is not None:
                    triggered = True
                    self.publisher_log.publish("[PostProcess] 收到抓取指令，开始自动寻优...")
            except Exception:
                pass  # 没收到指令是正常状态，静默跳过

            # 2. 获取最新视觉战报
            detect_res = None
            try:
                detect_res = self.subscriber_detect.get_message(timeout=0.001)
            except Exception:
                pass

                # 未触发，直接让出 CPU 切片，防止卡死 UI
            if not triggered:
                time.sleep(0.03)
                continue

            if detect_res is None or len(detect_res) == 0:
                self.publisher_log.publish("[PostProcess] 触发失败：当前视野内无目标")
                continue

            # ---------------------------------------------------------
            # 第二阶段：目标特征提取与优选算法
            # ---------------------------------------------------------
            valid_balls = []
            cubes = []

            # 遍历解析 YOLO 矩阵 (格式: [x1, y1, x2, y2, conf, cls_id])
            for det in detect_res:
                x1, y1, x2, y2, conf, cls_id = det
                cls_id = int(cls_id)
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                if cls_id == 0:
                    cubes.append([cx, cy])  # 记录魔方障碍物
                    continue

                if 1 <= cls_id <= 6:
                    # 计算欧氏距离的平方 (节省开方运算性能)
                    dist_to_center = (cx - center_x) ** 2 + (cy - center_y) ** 2
                    valid_balls.append({
                        "cls_id": cls_id,
                        "center": [cx, cy],
                        "dist": dist_to_center
                    })

            if not valid_balls:
                self.publisher_log.publish("[PostProcess] 决策终止：未检测到有效台球")
                continue

            # 🔥 [核心代码] 降维打击的双重排序
            # 优先级：1. 分数最高 (-cls_id 越小越排前)；2. 距中心最近 (dist 越小越排前)
            valid_balls.sort(key=lambda x: (-x["cls_id"], x["dist"]))
            best_target = valid_balls[0]

            # ---------------------------------------------------------
            # 第三阶段：2D 向量势场避障
            # ---------------------------------------------------------
            target_cx, target_cy = best_target["center"]
            safe_distance = 60  # 危险阈值 (像素)
            push_strength = 25  # 排斥偏移量 (像素)

            for cube_cx, cube_cy in cubes:
                dx = target_cx - cube_cx
                dy = target_cy - cube_cy
                dist = (dx ** 2 + dy ** 2) ** 0.5

                if 0 < dist < safe_distance:
                    ux = dx / dist
                    uy = dy / dist
                    # 顺着斥力方向偏移目标点
                    target_cx += ux * push_strength
                    target_cy += uy * push_strength
                    self.publisher_log.publish(
                        f"⚠️ 触发避障: 探测到魔方阻挡，坐标已偏移 ({ux * push_strength:.1f}, {uy * push_strength:.1f})"
                    )

            # 更新为安全抓取坐标
            best_target["center"] = [target_cx, target_cy]

            # ---------------------------------------------------------
            # 第四阶段：物理空间解算与指令下发
            # ---------------------------------------------------------
            try:
                # 像素坐标 (Pixel) -> 基座 3D 坐标 (Base) -> 极坐标 (Polar)
                base_pos = self.position_calculator.pix2base(best_target["center"], "ball")
                polar_pos = self.position_calculator.base2polar(base_pos)

                rho = float(polar_pos[0])
                theta = float(polar_pos[1])

                # 依据协议封装数据包：[Command, rho, theta]
                cmd_data = [0, rho, theta]
                self.publisher.publish(cmd_data)

                # 完美完成一次击杀
                log_msg = f"🟢 锁定目标 ID:{best_target['cls_id']} | 极径 R:{rho:.1f}cm | 角度 Theta:{theta:.2f}rad"
                self.publisher_log.publish(log_msg)
                print(f"[PostProcess] {log_msg}")

            except Exception as e:
                err_msg = f"❌ 坐标解算异常: {str(e)}"
                self.publisher_log.publish(err_msg)
                print(err_msg)

            time.sleep(0.02)

    def end(self) -> None:
        """
        优雅退出 (Graceful Shutdown) 的工业级实现。
        不仅修改标志位，还要通知 Qt 底层安全销毁线程事件循环。
        """
        self.stop = True
        self.quit()  # 退出事件循环
        self.wait()  # 阻塞等待线程真正结束，防止野线程导致内存泄漏