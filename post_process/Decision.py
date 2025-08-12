import Bridge, Debug, Position_calculate
import numpy as np
import time



# 这里是定义了一个目标类
class Target:
    def __init__(self, yolov5_res_single:np.array, position:np.array) -> None:
        self.pix = np.array([yolov5_res_single[0] + yolov5_res_single[2], yolov5_res_single[1] + yolov5_res_single[3]]) / 2 # 目标的中心点像素坐标
        self.centre = position  # 行向量
        # self.area = (yolov5_res_single[2] - yolov5_res_single[0]) * (yolov5_res_single[1] - yolov5_res_single[3])
        self.confidence = yolov5_res_single[4] 
        self.final_confidence = 0
        from Post_process import Mean_filter
        self.average_confidence = Mean_filter(5) # 置信度平均滤波器
        self.average_space = Mean_filter(5)
        self.average_crow = Mean_filter(5)
        self.average_edge = Mean_filter(5)
        self.type = yolov5_res_single[5]  # 0: cube 1: yellow 2: green 3: brown 4: blue 5: pink 6: black
        self.type_score = 0
        self.time_score = 0  # 时间间隔得分
        self.total_score = 0
        self.is_matched = False
        self.is_target = False

class Decision_maker():
    def __init__(self, config:dict, debuger:Debug.Debuger):
        self.match_distance = float(config["match_distance"])
        self.score_decrease_rate = float(config["score_decrease_rate"])
        self.score_increase_rate = float(config["score_increase_rate"])
        self.arm_length = float(config["arm_length"])
        self.safe_dis = float(config["safe_dis"])
        self.balls_score_weight = np.array(config["balls_score_weight"], dtype=np.float16)
        self.cubes_score_weight=np.array(config["cubes_score_weight"], dtype=np.float16)
        self.score_bias = float(config["score_bias"])
        self.image_shape = config["image_shape"]
        self.edge_threshold = config["edge_threshold"]
        self.bigbox_R = config["bigbox_R"]

        self.balls:list[Target] = []
        self.cubes:list[Target] = []
        self.box = Target(np.array(np.zeros((6))), np.array(np.zeros((3))))
        self.big = Target(np.array(np.zeros((6))), np.array(np.zeros((3))))
        self.debuger = debuger

        self.last_update_time = time.time() # 将当前的时间戳赋值给变量self.last_update_time，以便于跟踪上次更新的时间，计算时间间隔

        self.sampling_pts = np.zeros((9, 4))
        id = 1
        for r in [10, 20]:
            for theta in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                self.sampling_pts[id, :2] = [r * np.cos(theta), r * np.sin(theta)]
                id += 1

        self.ball_pub = Bridge.Publisher("ball_pix")
        self.cube_pub = Bridge.Publisher("cube_pix")

    def update(self, targets:list[Target]):
        # 老目标分数的自然衰减，时间依赖的得分减少行为
        dt = time.time() - self.last_update_time
        for index in range(len(self.balls)):
            self.balls[index].time_score -= self.score_decrease_rate * dt
        for index in range(len(self.cubes)):
            self.cubes[index].time_score -= self.score_decrease_rate * dt
        if not self.box is None:
            self.box.time_score -= self.score_decrease_rate * dt

        new_balls:list[Target] = []
        new_cubes:list[Target] = []
        new_box:list[Target] = []
        new_big:list[Target] = []

        # 对目标进行分类
        for target in targets:
            if int(target.type) == 0:
                new_cubes.append(target)
            elif int(target.type) in range(1, 7):
                new_balls.append(target)
            elif int(target.type) in range(7, 13):
                new_box.append(target)
            elif int(target.type) == 13:
                new_big.append(target)
            else:
                print(f"unkonw id:{target.type}")

        # 上一帧目标匹配标志位清零
        for i in range(len(self.balls)):
            self.balls[i].is_matched = False
        for i in range(len(self.cubes)):
            self.cubes[i].is_matched = False
        
        # 与上一帧匹配目标
        for index in range(len(new_balls)):
            # 按远近排序，使较近的优先匹配
            sorted(self.balls, key=lambda old: np.linalg.norm(old.centre - new_balls[index].centre))
            for ball_index in range(len(self.balls)):
                if new_balls[index].type == self.balls[ball_index].type:
                    if np.linalg.norm(np.array(new_balls[index].centre - self.balls[ball_index].centre)) < self.match_distance:
                        self.balls[ball_index].time_score += self.score_increase_rate * dt
                        if self.balls[ball_index].time_score > 1:
                            self.balls[ball_index].time_score = 1
                        self.balls[ball_index].centre = new_balls[index].centre
                        self.balls[ball_index].pix = new_balls[index].pix
                        new_balls[index].is_matched = True
                if self.balls[ball_index].is_matched:
                    continue

        
        for index in range(len(new_cubes)):
            # 按远近排序，使较近的优先匹配
            sorted(self.cubes, key=lambda old: np.linalg.norm(old.centre - new_cubes[index].centre))
            for cube_index in range(len(self.cubes)):
                # cube就只有一类，没必要进行类型确认
                if np.linalg.norm(np.array(new_cubes[index].centre - self.cubes[cube_index].centre)) < self.match_distance:
                    self.cubes[cube_index].time_score += self.score_increase_rate * dt
                    if self.cubes[cube_index].time_score > 1:
                            self.cubes[cube_index].time_score = 1
                    self.cubes[cube_index].centre = new_cubes[index].centre
                    self.cubes[cube_index].pix = new_cubes[index].pix
                    new_cubes[index].is_matched = True

        
        if len(new_box) > 0:
            if not self.box.is_matched:
                # 第一次发现box由置信度来选
                new_box = sorted(new_box, key=lambda x: x.confidence)
                self.box = new_box[-1]
                self.box.is_matched = True
            else:
                # 取最近的进行匹配
                new_box = sorted(new_box, key=lambda x: np.linalg.norm(x.centre - self.box.centre))
                self.box.centre = new_box[0].centre
                self.box.pix = new_box[0].pix
            self.box.time_score += self.score_increase_rate * dt
            if self.box.time_score > 1:
                self.box.time_score = 1
            
        if len(new_big) > 0:
            if not self.big.is_matched:
                # 第一次发现box由置信度来选
                new_big = sorted(new_big, key=lambda x: x.confidence)
                self.big = new_big[-1]
                self.big.is_matched = True
            else:
                # 取最近的进行匹配
                new_big = sorted(new_big, key=lambda x: np.linalg.norm(x.centre - self.box.centre))
                self.big.centre = new_big[0].centre
                self.big.pix = new_big[0].pix
            self.big.time_score += self.score_increase_rate * dt
            if self.big.time_score > 1:
                self.big.time_score = 1

        # 对新出现的目标进行处理 
        unmatch_balls = 0
        for index in range(len(new_balls)):
            if new_balls[index].is_matched:
                continue

            new_balls[index].time_score = self.score_increase_rate * dt
            # 读取图中已有的物体并计算得分（已经计算的不用重复计算）
            new_balls[index].type_score = 0.2 * (new_balls[index].type - 1) # 计算小球的得分
            self.balls.append(new_balls[index])
            unmatch_balls += 1
        
        for index in range(len(new_cubes)):
            if new_cubes[index].is_matched:
                continue

            new_cubes[index].time_score = self.score_increase_rate * dt
            new_cubes[index].type_score = 0
            self.cubes.append(new_cubes[index])
            
        # 判断物体所在范围对抓取的影响进而来判断位置得分，希望定义一个函数来线性计算
        for index in range (len(self.balls)):
            # 每次更新都需要重新计算，再套滤波器
            self.balls[index].average_space.push(self.scope_scores(self.balls[index].centre[0:2]))
            self.balls[index].average_confidence.push(self.balls[index].confidence) #  平均值滤波
            self.balls[index].average_edge.push(self.get_edge_score(self.balls[index].centre))
            crowd_score = 0
            self.balls[index].crowd_list=[]
            # 认为只有魔方会挡着抓取
            for inner_id in range (len(self.cubes)):
                dis = np.linalg.norm(self.balls[index].centre[0:2] - self.cubes[inner_id].centre[0:2])
                self.balls[index].crowd_list.append(dis)
                self.balls[index].crowd_list.sort(reverse=False)
                if self.balls[index].crowd_list[0] >= 8:
                    crowd_score  = 1
                else:
                    crowd_score = 0
                self.balls[index].average_crow.push(crowd_score)
            if self.is_edge_target(self.balls[index].pix):
                self.balls[index].average_edge.push(-1)
            else:
                self.balls[index].average_edge.push(0)
            self.balls[index].average_crow.push(crowd_score)
            if self.balls[index].average_space.get_average() < 0:
                self.balls[index].total_score = -np.inf
            else:
                self.balls[index].total_score = (self.balls_score_weight[0] * self.balls[index].average_confidence.get_average() +
                                                self.balls_score_weight[1] * self.balls[index].type_score +
                                                self.balls_score_weight[2] * self.balls[index].time_score +
                                                self.balls_score_weight[3] * self.balls[index].average_crow.get_average() +
                                                self.balls_score_weight[4] * self.balls[index].average_edge.get_average())
                if self.balls[index].is_target:
                    self.balls[index].total_score += self.balls_score_weight[5]
        for index in range (len(self.cubes)):
            self.cubes[index].average_space.push(self.scope_scores(self.cubes[index].centre[0:2])) #  通过置信度来判断得分，希望对置信度做滤波处理
            self.cubes[index].average_confidence.push(self.cubes[index].confidence) #  平均值滤波
            self.cubes[index].average_edge.push(self.get_edge_score(self.cubes[index].centre))
            self.cubes[index].cube_crowd=[]
            crowd_score=0
            for inner_id in range (len(self.cubes)):
                if inner_id == index:
                    continue
                dis = np.linalg.norm(self.cubes[index].centre[0:2] - self.cubes[inner_id].centre[0:2])
                self.cubes[index].cube_crowd.append(dis)
                self.cubes[index].cube_crowd.sort(reverse=False)
                if self.cubes[index].cube_crowd[0] >= 8:
                    crowd_score = 1
                else:
                    crowd_score = 0
                self.cubes[index].average_crow.push(crowd_score)
            if self.is_edge_target(self.cubes[index].pix):
                self.cubes[index].average_edge.push(-1)
            else:
                self.cubes[index].average_edge.push(0)
            if self.cubes[index].average_space.get_average() < 0:
                self.cubes[index].total_score = -np.inf
            else:
                self.cubes[index].total_score = (self.cubes_score_weight[0] * self.cubes[index].average_confidence.get_average() +
                                                self.cubes_score_weight[1] * self.cubes[index].type_score +
                                                self.cubes_score_weight[2] * self.cubes[index].time_score +
                                                self.cubes_score_weight[3] * self.cubes[index].average_crow.get_average() +
                                                self.cubes_score_weight[4] * self.cubes[index].average_edge.get_average())
                if self.cubes[index].is_target:
                    self.cubes[index].total_score += self.cubes_score_weight[5]
        # 去除评分过低的目标
        delet_list = []
        for index in range(len(self.balls)):
            if self.balls[index].time_score < 0:  # 设置删除阈值分数
                delet_list.append(index)
        delet_list.reverse() # 列表里元素的翻转
        for index in delet_list:
            del self.balls[index] # 删除列表里相应的元素
        delet_list.clear()
        for index in range(len(self.cubes)):
            if self.cubes[index].time_score < 0: # 设置删除阈值分数
                delet_list.append(index)
        delet_list.reverse()
        for index in delet_list:
            del self.cubes[index]
        if self.box.total_score < 0:
            self.box.is_matched = False
        if self.big.total_score < 0:
            self.box.is_matched = False

        # 排序算法的降序排列使分最高的为第一个
        self.balls.sort(key=lambda target : target.total_score, reverse=True)
        self.cubes.sort(key=lambda target : target.total_score, reverse=True)

        if self.balls:
            if not self.balls[0].is_target:
                for i in range(len(self.balls)):
                    self.balls[i].is_target = False
                self.balls[0].is_target = True

        if self.cubes:
            if not self.cubes[0].is_target:
                for i in range(len(self.cubes)):
                    self.cubes[i].is_target = False
                self.cubes[0].is_target = True
        
        if len(self.balls) != 0:
            self.ball_pub.publish(self.balls[0].pix)
        else:
            self.ball_pub.publish(None)
        if len(self.cubes) != 0:
            self.cube_pub.publish(self.cubes[0].pix)
        else:
            self.cube_pub.publish(None)

        self.last_update_time = time.time()
        
    def scope_scores(self, xy): # 传入物体所在的坐标范围，x坐标，y坐标 进行线性赋分，返回得分值
        distance = np.linalg.norm(xy)
        score = 1 - distance / self.arm_length
        return score
    
    def get_edge_score(self, tar_position):
        if self.big.is_matched == False:
            return 0
        else:
            distance = np.linalg.norm(tar_position - self.big.centre)
            return 1 - distance / (self.bigbox_R)
                     
    def get_ball_base(self):
        if len(self.balls) == 0:
            raise Exception("No ball detected")
        return self.balls[0].centre
    
    def get_cube_base(self):
        if len(self.cubes) == 0:
            raise Exception("No cube detected")
        return  self.cubes[0].centre
    
    def get_box_base(self):
        if not self.box.is_matched:
            raise Exception("No box detected")
        return self.box.centre
    
    def get_big_base(self):
        if not self.big.is_matched:
            raise Exception("No big detected")
        return self.big.centre
    
    def get_open_area(self) -> np.array:
        '''
        return: 一个开阔地区的行向量
        '''
        pts = self.sampling_pts.copy()
        for i in range(len(pts)):
            for cube in self.cubes:
                pts[i, 3] += np.linalg.norm(cube.centre - pts[i, :3])
        pts = sorted(pts, key=lambda x : x[3])
        return np.array(pts[0][:3])
    
    def is_edge_target(self, pt):
        if (self.edge_threshold < pt[0] < self.image_shape[0] - self.edge_threshold and
            self.edge_threshold < pt[1] < self.image_shape[1] - self.edge_threshold):
            return False
        else:
            return True
