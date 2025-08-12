import numpy as np
import os, toml, cv2

class Position_calculator():
    def __init__(self, debuger):
        super().__init__()

        # 很普通的造字典
        ROOT = 	os.getcwd()
        config = toml.load(os.path.join(ROOT, "config/config.toml"))
        specific_config = config["specific_config"]
        PATH = os.path.join(ROOT, "config", specific_config)
        PATH = PATH.replace('/', os.sep).replace("\\", os.sep) # 确保路径字符串中的路径分隔符在不同操作系统下都是一致的，以便正确地处理路径。
        config = toml.load(PATH)

        # 字典赋值
        self.K = np.array(config["camera"]['K'], dtype=np.float32)
        self.D = np.array(config["camera"]["D"], dtype=np.float32)
        self.alpha = config["camera"]["alpha"]
        self.image_shape = config["camera"]["image_shape"]
        self.new_K, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, self.image_shape, self.alpha, self.image_shape, True)
        self.camera_height = config["position_calculate"]["camera_height"]
        # 解包config里的bc_vec，使之成为一个新的类型为float的列表，并将其转置
        self.c2b_vec = np.array(config["position_calculate"]["c2b_vec"], dtype=np.float32).reshape((3, 1))# ?
        # 体心高度
        self.types_dis = {"floor":self.camera_height, "cube":self.camera_height - config["position_calculate"]["cube_height"],
                             "ball":self.camera_height - config["position_calculate"]["ball_height"] / 2,
                             "box":self.camera_height - config["position_calculate"]["box_height"]}
        # 机械臂的倾斜角度
        self.arm_tilt  = np.array(config["position_calculate"]["arm_tilt"] , dtype=np.float32) / 180 * np.pi
        # 盒子半径
        self.bigbox_R = config["position_calculate"]["bigbox_R"]
        #
        self.edge_threshold = config["position_calculate"]["edge_threshold"]
        self.arm_length = config["decision"]["arm_length"]

        # 得到一个旋转矩阵（主要任务是用来矫正机械的锅
        c2b_rotation_x = np.array([1, 0, 0]).reshape((3, 1)) * self.arm_tilt[0]
        c2b_rotation_x = cv2.Rodrigues(c2b_rotation_x)[0]
        c2b_rotation_y = np.array([0, 1, 0]).reshape((3, 1)) * self.arm_tilt[1]
        c2b_rotation_y = cv2.Rodrigues(c2b_rotation_y)[0]
        c2b_rotation_z = np.array([0, 0, 1]).reshape((3, 1)) * self.arm_tilt[2]
        c2b_rotation_z = cv2.Rodrigues(c2b_rotation_z)[0]
        # self.c2b_rotation = c2b_rotation_z @ c2b_rotation_y @ c2b_rotation_x
        print(f"{c2b_rotation_x = }")
        print(f"{c2b_rotation_y = }")
        print(f"{c2b_rotation_z = }")
        T_BA = np.array(config["position_calculate"]["T_BA"], dtype=np.float32)
        self.c2b_rotation = T_BA[:3,:3]

        self.debuger = debuger

    def pix2base(self, centre_point:list, type:str) -> np.array:
        # 定义pix系、camera系、base系的x轴和y轴都是分别平行的
        # 解包操作
        pix_point = np.array([*centre_point, 1]).reshape((3, 1))
        # 计算相机坐标系的点
        camera_point_ = np.linalg.inv(self.new_K) @ pix_point
        # 计算深度（？
        # ！！！！！！！！！
        Z_c = (self.types_dis[type] - self.c2b_rotation[2, :] @ self.c2b_vec) / (self.c2b_rotation[2, :] @ camera_point_)

        object_point = self.c2b_rotation @ (camera_point_ * Z_c + self.c2b_vec)
        return object_point
    # 极坐标转换为基座标下的点
    def polar2base(self, polar_point, type):
        vec = self.c2b_rotation @ np.array([[0], [1], [0]]) * polar_point[0]
        R = np.linalg.norm(vec[:2, 0])
        base_pt = np.array([R * np.cos(polar_point[1]),
                            R * np.sin(polar_point[1]),
                            self.types_dis[type]])
        return base_pt

    # 将基坐标转换为极坐标
    def base2polar(self, base_point:np.array):
        # 定义极轴为相机到滑环的方向，角度的正方向为俯视时的顺时针
        # 经测试代码只要将这个定义下的角度发给电控就能把机械臂转到正确的位置
        # 角度为弧度制，长度用cm
        # where极点？
        bp = base_point.reshape((3, 1))
        arm_point = np.linalg.inv(self.c2b_rotation) @  bp
        R = np.linalg.norm(arm_point[:2, 0])
        theta = np.arctan2(bp[1, 0], bp[0, 0])
        theta += np.pi / 2
        # 限幅，让角度限在半圈内
        if theta >= np.pi:
            theta -= 2 * np.pi
        return np.array([R, theta])
    # 基坐标转换为像素坐标
    def base2pix(self, base_point):
        base = np.array(base_point).reshape((3, 1))
        camera_point = np.linalg.inv(self.c2b_rotation) @  base - self.c2b_vec
        pix_point = self.new_K @ (camera_point / camera_point[2, 0])
        return (np.array(pix_point[0:2]).reshape(2))
    # 获取基座原点在像素坐标系中的位置
    def get_base_origin_in_pix(self, type):
        base_point = np.array([0, 0, self.types_dis[type]]).reshape((3, 1))
        return self.base2pix(base_point)
    # 给出中心坐标
    def bigbox_get_base(self, target_xyxy):
        # leftup左上 rightdown右下
        lu, rd = target_xyxy[:2], target_xyxy[2:]
        # 对应在地板上的坐标
        lu_base = self.pix2base(lu, "floor")
        rd_base = self.pix2base(rd, "floor")
        cen_base = self.pix2base([self.image_shape[0] / 2, self.image_shape[1] / 2], "floor")
        # if (lu[0] < self.edge_threshold):
        #     if lu[1] < self.edge_threshold:
        #         if rd[0] < self.image_shape[0] - self.edge_threshold:
        #             if rd[1] < self.image_shape[1] - self.edge_threshold:
        #                 centre_base = rd_base - np.array([[self.bigbox_R], [self.bigbox_R], [self.camera_height]])
        #             else:
        #                 centre_base = np.array([[rd_base[0, 0] - self.bigbox_R], [cen_base[1, 0]], [self.camera_height]])
        #         else:
        #             if rd[1] < self.image_shape[1] - self.edge_threshold:
        #                 centre_base = np.array([[centre_base[0, 0]], [rd_base[1, 0] - self.bigbox_R], [self.camera_height]])
        #             else:
        #                 centre_base = cen_base  # 其实不太可能
        #     else:
        #         if rd[0] < self.image_shape[0] - self.edge_threshold:
        #             if rd[1] < self.image_shape[1] - self.edge_threshold:
        #                 cen_base = np.array([[rd_base[0, 0] - self.bigbox_R], [(lu_base[1, 0] + rd_base[1, 0]) / 2], [self.camera_height]])
        #             else:
        #                 cen_base = np.array([[rd_base[0, 0] - self.bigbox_R], [lu_base[0, 0] + self.bigbox_R], [self.camera_height]])
        #         else:
        #             if rd[1] < self.image_shape[1] - self.edge_threshold:
        #                 cen_base = np.array([[cen_base[0, 0]], [(lu_base[1, 0] + rd_base[1, 0]) / 2], [self.camera_height]])  # 也不太可能
        #             else:
        #                 cen_base = np.array([[cen_base[0, 0]], [lu_base[1, 0] + self.bigbox_R], [self.camera_height]])
        # else:
        #     if lu[1] < self.edge_threshold:
        #         if rd[0] < self.image_shape[0] - self.edge_threshold:
        #             if rd[1] < self.image_shape[1] - self.edge_threshold:
        #                 cen_base = np.array([[(lu_base[0] + rd_base[0]) / 2], [rd_base[1] - self.bigbox_R], [self.camera_height]])
        #             else:
        #                 cen_base = np.array([[(lu_base[0] + rd_base[0]) / 2]]])
        # 我好像明白了什么。。。
        # 当目标物体横跨图像左右边缘时，直接使用图像中心在基座坐标系中的 x 坐标作为中心点的 x 坐标。
        # 当目标物体左边界靠近图像左边缘但右边界正常时，用右下角的 x 坐标减去大盒子半径作为中心点的 x 坐标。
        # 当目标物体右边界靠近图像右边缘但左边界正常时，用左上角的 x 坐标加上大盒子半径作为中心点的 x 坐标。
        # 当目标物体左右边界都正常时，使用左上角和右下角的 x 坐标的平均值作为中心点的 x 坐标。
        if lu[0] < self.edge_threshold:
            if rd[0] > self.image_shape[0] - self.edge_threshold:
                x_base = cen_base[0, 0]
            else:
                x_base = rd_base[0, 0] - self.bigbox_R
        else:
            if rd[0] > self.image_shape[0] - self.edge_threshold:
                x_base = lu_base[0, 0] + self.bigbox_R
            else:
                x_base = (lu_base[0, 0] + rd_base[0, 0]) / 2
        if lu[1] < self.edge_threshold:
            if rd[1] > self.image_shape[1] - self.edge_threshold:
                y_base = cen_base[1, 0]
            else:
                y_base = rd_base[1, 0] - self.bigbox_R
        else:
            if rd[1] > self.image_shape[1] - self.edge_threshold:
                y_base = lu_base[1, 0] + self.bigbox_R
            else:
                y_base = (lu_base[1, 0] + rd_base[1, 0]) / 2
        return np.array([[x_base], [y_base], [self.camera_height]])
    # 360度转动时候机械臂所在的像素坐标，切片了500个数据（还要判断这个点是否在图像里边
    def get_rotation_elliptic_pts(self, type, im_shape):
        pts = []
        for theta in np.linspace(0, 2 * np.pi, 500):
            pt = np.array(self.base2pix(self.polar2base([self.arm_length, theta], type)), dtype=np.uint16)
            if 0 <= pt[0] < im_shape[0] and 0 <= pt[1] < im_shape[1]:
                pts.append(pt)
        return np.array(pts, dtype=np.uint16)