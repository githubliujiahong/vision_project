import Bridge, Debug, Position_calculate
import numpy as np
import time
from Post_process import Mean_filter
# from Decision import Target

num2type = ["cube", "ball", "ball", "ball", "ball", "ball", "ball", "box", "box", "box", "box", "box", "box", "big_box"]

class The_last_dec():
    def __init__(self,the_best,balls,cubes):
        self.best=the_best
        self.balls=balls
        self.cubes=cubes
        self.the_near_dis = 3
        self.need_pix = None
        self.Intermediate_quantity_x = []
        self.Intermediate_quantity_y = []
        self.position_calculator = Position_calculate.Position_calculator(Debug.Debuger())
    def decide(self):
        the_near_obj=[]
        for ball in self.balls:
            if (ball.centre[0]-self.best.centre[0])**2+(ball.centre[1]-self.best.centre[1])**2<self.the_near_dis**2:
                the_near_obj.append(ball)
        for cube in self.cubes:
            if (cube.centre[0]-self.best.centre[0])**2+(cube.centre[1]-self.best.centre[1])**2<self.the_near_dis**2:
                the_near_obj.append(cube)
        if not the_near_obj:
            return         
        for obj in the_near_obj:
            the_x = obj.pix[0]-self.best.pix[0]
            the_y = obj.pix[1]-self.best.pix[1]
            the_x = self.best.pix[0]-the_x
            the_y = self.best.pix[1]-the_y
            self.Intermediate_quantity_x.append(the_x)
            self.Intermediate_quantity_y.append(the_y)
        if self.Intermediate_quantity_x and self.Intermediate_quantity_y:
            avg_x = sum(self.Intermediate_quantity_x) / len(self.Intermediate_quantity_x)
            avg_y = sum(self.Intermediate_quantity_y) / len(self.Intermediate_quantity_y)
            self.need_pix = [int(avg_x), int(avg_y)]
            self.best.pix = self.need_pix  
            self.best.centre = self.position_calculator.pix2base(self.need_pix, self.best.type)
        else:
            print("Maybe MAN")
         # ！！！！！！！！！          








