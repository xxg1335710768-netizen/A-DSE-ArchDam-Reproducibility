import math
import numpy as np
import re
from scipy.optimize import fsolve
from scipy.stats import qmc  # 引入LHS采样模块
import time


def parabola_arc_length(x1, b):
    return x1 * (1 + (1 / 10) * (2 * x1 / b) ** 2)


def para_type2(x, y, phi2, model='xy'):  # 求解对数螺旋线参数
    def equations(vars):
        rho0, k = vars
        beta = math.atan(k)
        phi = phi2 * math.pi / 180
        eq1 = x - (rho0 * math.exp(k * phi) * math.sin(beta + phi) - rho0 * math.sin(beta))
        if model == 'xy':
            eq2 = y - (rho0 * math.exp(k * phi) * math.cos(beta + phi) - rho0 * math.cos(beta))
        else:
            eq2 = y - rho0 * math.sqrt(1 + k ** 2)
        return [eq1, eq2]

    solutions = fsolve(equations, [100, 0.1])
    rho0, k = solutions
    beta = math.atan(k)
    return rho0, k, beta


class DamParameter:
    def __init__(self, Type=0, Z_top=2540, Z_up=2535, Z_bot=2305,
                 Z_down=2352, Z_sand=2420, Sand_dens=8000,
                 Sand_angle=12, Layer_num=10, Hole_num=9, K_z1=0.4,
                 Sup_num=3, Hengfeng_num=0, Hengfeng_r=293, Hengfeng_d=1,
                 Hengfeng_range=95, Hengfeng_begin=47.85, BC_h=1.5, Y_offset=20,
                 Edge_xl=2.5, Edge_xr=2.5, Edge_yu=1.5, Edge_yd=1.5, Edge_base=1.5,
                 Slope_ptnum=50, Gong_ptnum=11, T=1):
        self.Type = Type
        self.Z_top = Z_top
        self.Z_up = Z_up
        self.Z_bot = Z_bot
        self.Z_down = Z_down
        self.Z_sand = Z_sand
        self.Sand_dens = Sand_dens
        self.Sand_angle = Sand_angle
        self.Layer_num = Layer_num
        self.Hole_num = Hole_num
        self.K_z1 = K_z1
        self.Sup_num = Sup_num
        self.Hengfeng_num = Hengfeng_num
        self.Hengfeng_r = Hengfeng_r
        self.Hengfeng_d = Hengfeng_d
        self.Hengfeng_range = Hengfeng_range
        self.Hengfeng_begin = Hengfeng_begin
        self.Hengfeng_angle = Hengfeng_range / (Hengfeng_num - 1) if Hengfeng_num > 1 else 0
        self.BC_h = BC_h
        self.Y_offset = [0, Y_offset, 0]
        self.Edge_xl = Edge_xl * (Z_top - Z_bot)
        self.Edge_xr = -Edge_xr * (Z_top - Z_bot)
        self.Edge_yu = Edge_yu * (Z_top - Z_bot)
        self.Edge_yd = -Edge_yd * (Z_top - Z_bot)
        self.Edge_base = Edge_base * (Z_top - Z_bot)
        self.Slope_ptnum = Slope_ptnum
        self.Gong_ptnum = Gong_ptnum
        self.T = T

    def InitialParameter(self):
        self.Gong_h = [2540, 2510, 2490, 2470, 2440, 2410, 2380, 2350, 2325, 2305]
        self.Y_up = [0, 9.3591, 18.1767, 24.1623, 28.1644, 29.3429, 28.382, 25.9659, 23.3345, 21.1099]
        self.T_m = [11, 19.641, 27.914, 33.86, 38.654, 41.738, 44.009, 46.361, 49.029, 52]
        self.T_l = [15, 21.866, 29.088, 34.961, 40.467, 44.575, 47.582, 49.786, 51.223, 52.236]
        self.T_r = [15, 21.708, 29.426, 36.299, 43.278, 48.694, 52.356, 54.075, 53.886, 52.595]
        self.R_l = [245.432, 225.527, 202.902, 182.696, 161.511, 143.515, 128.546, 116.442, 108.428, 103.305]
        self.R_r = [243.863, 224.584, 204.336, 187.502, 170.502, 155.481, 140.836, 124.966, 109.648, 95.356]
        self.X_l = [231.1448, 221.6624, 207.3984, 193.7154, 172.7808, 150.1109, 124.5561, 94.7071, 61.6605, 27.7829]
        self.X_r = [229.1375, 222.6391, 211.5801, 199.2147, 182.8207, 161.5395, 128.6544, 93.16, 58.7186, 19.1873]
        self.Angle_l = [round(math.atan(self.X_l[i] / self.R_l[i]) * 180 / math.pi, 4) for i in range(self.Layer_num)]
        self.Angle_r = [round(math.atan(self.X_r[i] / self.R_r[i]) * 180 / math.pi, 4) for i in range(self.Layer_num)]
        self.Angle_l_ini = self.Angle_l[-1]
        self.Angle_r_ini = self.Angle_r[-1]
        self.Hole_x = [-32.38, -11.34, 9.5, 30.34, 51.38, -21.82, -0.3, 19.91, 40.83]
        self.Hole_y = [0] * 9
        self.Hole_z = [11.5, 11.5, 11.5, 11.5, 11.5, 81.75, 81.75, 81.75, 81.75]
        self.Hole_width = [10, 10, 10, 10, 10, 5, 5, 5, 5]
        self.Hole_height = [12.5, 12.5, 12.5, 12.5, 12.5, 6.5, 6.5, 6.5, 6.5]

        h = [i - self.Z_bot for i in self.Gong_h]
        m = [i / (self.Z_top - self.Z_bot) for i in h]

        Ym = np.polyfit(m, self.Y_up, 3)
        self.A1, self.B1, self.C1, self.D1 = Ym
        Tm = np.polyfit(m, self.T_m, 3)
        self.A2, self.B2, self.C2, self.D2 = Tm
        Rlm = np.polyfit(m, self.R_l, 3)
        self.A3, self.B3, self.C3, self.D3 = Rlm
        Rrm = np.polyfit(m, self.R_r, 3)
        self.A4, self.B4, self.C4, self.D4 = Rrm
        Tlm = np.polyfit(m, self.T_l, 3)
        self.A5, self.B5, self.C5, self.D5 = Tlm
        Trm = np.polyfit(m, self.T_r, 3)
        self.A6, self.B6, self.C6, self.D6 = Trm
        Xlm = np.polyfit(m, self.X_l, 3)
        self.A7, self.B7, self.C7, self.D7 = Xlm
        Xrm = np.polyfit(m, self.X_r, 3)
        self.A8, self.B8, self.C8, self.D8 = Xrm
        Anglelm = np.polyfit(m, self.Angle_l, 3)
        self.A9, self.B9, self.C9, self.D9 = Anglelm
        Anglerm = np.polyfit(m, self.Angle_r, 3)
        self.A10, self.B10, self.C10, self.D10 = Anglerm

        self.kt_initial = 3 * self.A1 / (self.Z_top - self.Z_bot) + 2 * self.B1 / (
                self.Z_top - self.Z_bot) + self.C1 / (self.Z_top - self.Z_bot)
        self.kb_initial = self.C1 / (self.Z_top - self.Z_bot)
        self.m0_initial = ((-self.B1 - math.sqrt(self.B1 ** 2 - 3 * self.A1 * self.C1)) * (self.Z_top - self.Z_bot)) / (
                3 * self.A1)
        self.n0_initial = self.A1 * (self.m0_initial / (self.Z_top - self.Z_bot)) ** 3 + self.B1 * (
                self.m0_initial / (self.Z_top - self.Z_bot)) ** 2 + self.C1 * (
                                  self.m0_initial / (self.Z_top - self.Z_bot)) + self.D1

        self.Tm_0_initial = self.D2
        self.Tm_1_initial = self.A2 / 27 + self.B2 / 9 + self.C2 / 3 + self.D2
        self.Tm_2_initial = 8 * self.A2 / 27 + 4 * self.B2 / 9 + 2 * self.C2 / 3 + self.D2
        self.Tm_3_initial = self.A2 + self.B2 + self.C2 + self.D2
        self.Tl_0_initial = self.D5
        self.Tl_1_initial = self.A5 / 27 + self.B5 / 9 + self.C5 / 3 + self.D5
        self.Tl_2_initial = 8 * self.A5 / 27 + 4 * self.B5 / 9 + 2 * self.C5 / 3 + self.D5
        self.Tl_3_initial = self.A5 + self.B5 + self.C5 + self.D5
        self.Tr_0_initial = self.D6
        self.Tr_1_initial = self.A6 / 27 + self.B6 / 9 + self.C6 / 3 + self.D6
        self.Tr_2_initial = 8 * self.A6 / 27 + 4 * self.B6 / 9 + 2 * self.C6 / 3 + self.D6
        self.Tr_3_initial = self.A6 + self.B6 + self.C6 + self.D6

        self.Anglel_0_initial = self.D9
        self.Anglel_1_initial = self.A9 / 27 + self.B9 / 9 + self.C9 / 3 + self.D9
        self.Anglel_2_initial = 8 * self.A9 / 27 + 4 * self.B9 / 9 + 2 * self.C9 / 3 + self.D9
        self.Anglel_3_initial = self.A9 + self.B9 + self.C9 + self.D9
        self.Angler_0_initial = self.D10
        self.Angler_1_initial = self.A10 / 27 + self.B10 / 9 + self.C10 / 3 + self.D10
        self.Angler_2_initial = 8 * self.A10 / 27 + 4 * self.B10 / 9 + 2 * self.C10 / 3 + self.D10
        self.Angler_3_initial = self.A10 + self.B10 + self.C10 + self.D10

        self.a_l, self.b_l, self.a_r, self.b_r, self.theta_l, self.theta_r = [], [], [], [], [], []
        if self.Type == 0:
            self.Y_l = [round(self.Y_up[i] - self.T_m[i] / 2 + self.X_l[i] ** 2 / (-2 * self.R_l[i]), 5) for i in
                        range(self.Layer_num)]
            self.Y_r = [round(self.Y_up[i] - self.T_m[i] / 2 + self.X_r[i] ** 2 / (-2 * self.R_r[i]), 5) for i in
                        range(self.Layer_num)]
        elif self.Type == 1:
            self.b_l = [-2 * self.R_l[i] for i in range(self.Layer_num)]
            self.a_l = [1 / math.tan(self.Angle_l[i] * math.pi / 180) ** 2 - self.b_l[i] ** 2 / (4 * self.X_l[i] ** 2)
                        for i in range(self.Layer_num)]
            self.b_r = [-2 * self.R_r[i] for i in range(self.Layer_num)]
            self.a_r = [1 / math.tan(self.Angle_r[i] * math.pi / 180) ** 2 - self.b_r[i] ** 2 / (4 * self.X_r[i] ** 2)
                        for i in range(self.Layer_num)]
            self.Y_l = [self.Y_up[i] - self.T_m[i] / 2 + (
                    -self.b_l[i] - math.sqrt(self.b_l[i] ** 2 + 4 * self.a_l[i] * self.X_l[i] ** 2) / 2 / self.a_l[
                i]) for i in range(self.Layer_num)]
            self.Y_r = [self.Y_up[i] - self.T_m[i] / 2 + (
                    -self.b_r[i] - math.sqrt(self.b_r[i] ** 2 + 4 * self.a_r[i] * self.X_r[i] ** 2) / 2 / self.a_r[
                i]) for i in range(self.Layer_num)]
        elif self.Type == 2:
            for i in range(self.Layer_num):
                a, b, theta = map(lambda v: round(v, 7), para_type2(self.X_l[i], self.R_l[i], self.Angle_l[i], 'xR'))
                self.a_l.append(float(a));
                self.b_l.append(float(b));
                self.theta_l.append(theta)
                a, b, theta = map(lambda v: round(v, 7), para_type2(self.X_r[i], self.R_r[i], self.Angle_r[i], 'xR'))
                self.a_r.append(float(a));
                self.b_r.append(float(b));
                self.theta_r.append(theta)
            self.Y_l = [
                self.Y_up[i] - self.T_m[i] / 2 - self.a_l[i] * math.cos(self.theta_l[i]) + self.a_l[i] * math.exp(
                    self.b_l[i] * self.Angle_l[i] * math.pi / 180) * math.cos(
                    self.theta_l[i] + self.Angle_l[i] * math.pi / 180) for i in range(self.Layer_num)]
            self.Y_r = [
                self.Y_up[i] - self.T_m[i] / 2 - self.a_r[i] * math.cos(self.theta_r[i]) + self.a_r[i] * math.exp(
                    self.b_r[i] * self.Angle_r[i] * math.pi / 180) * math.cos(
                    self.theta_r[i] + self.Angle_r[i] * math.pi / 180) for i in range(self.Layer_num)]

    def ParameterUpdate(self, para_list):
        self.a_l, self.b_l, self.a_r, self.b_r, self.theta_l, self.theta_r = [], [], [], [], [], []
        self.para_list = para_list
        h = [i - self.Z_bot for i in self.Gong_h]
        m = [i / (self.Z_top - self.Z_bot) for i in h]
        self.kt = self.kt_initial * self.para_list[0]
        self.kb = self.kb_initial * self.para_list[1]
        self.m0 = self.m0_initial * self.para_list[2]
        self.n0 = self.n0_initial * self.para_list[3]
        self.Tm_0 = self.Tm_0_initial * para_list[4]
        self.Tm_1 = self.Tm_1_initial * para_list[5]
        self.Tm_2 = self.Tm_2_initial * para_list[6]
        self.Tm_3 = self.Tm_3_initial * para_list[7]
        self.Tl_0 = self.Tl_0_initial * para_list[8]
        self.Tl_1 = self.Tl_1_initial * para_list[9]
        self.Tl_2 = self.Tl_2_initial * para_list[10]
        self.Tl_3 = self.Tl_3_initial * para_list[11]
        self.Tr_0 = self.Tr_0_initial * para_list[12]
        self.Tr_1 = self.Tr_1_initial * para_list[13]
        self.Tr_2 = self.Tr_2_initial * para_list[14]
        self.Tr_3 = self.Tr_3_initial * para_list[15]

        self.A1 = (self.Z_top - self.Z_bot) ** 2 * (self.m0 * (self.kt - self.kb) + self.kb *
                                                    (self.Z_top - self.Z_bot)) / (
                          3 * self.m0 * (self.Z_top - self.Z_bot - self.m0))
        self.B1 = -(self.kb * (self.Z_top - self.Z_bot) ** 3 + self.m0 ** 2 * (self.Z_top - self.Z_bot) *
                    (self.kt - self.kb)) / (2 * self.m0 * (self.Z_top - self.Z_bot - self.m0))
        self.C1 = self.kb * (self.Z_top - self.Z_bot)
        self.D1 = self.n0 - self.A1 * (self.m0 / (self.Z_top - self.Z_bot)) ** 3 - self.B1 * (self.m0 /
                                                                                              (
                                                                                                      self.Z_top - self.Z_bot)) ** 2 - self.C1 * self.m0 / (
                          self.Z_top - self.Z_bot)

        self.A2 = -9 / 2 * self.Tm_0 + 27 / 2 * self.Tm_1 - 27 / 2 * self.Tm_2 + 9 / 2 * self.Tm_3
        self.B2 = 9 * self.Tm_0 - 45 / 2 * self.Tm_1 + 18 * self.Tm_2 - 9 / 2 * self.Tm_3
        self.C2 = -11 / 2 * self.Tm_0 + 9 * self.Tm_1 - 9 / 2 * self.Tm_2 + self.Tm_3
        self.D2 = self.Tm_0

        self.A5 = - 9 / 2 * self.Tl_0 + 27 / 2 * self.Tl_1 - 27 / 2 * self.Tl_2 + 9 / 2 * self.Tl_3
        self.B5 = 9 * self.Tl_0 - 45 / 2 * self.Tl_1 + 18 * self.Tl_2 - 9 / 2 * self.Tl_3
        self.C5 = -11 / 2 * self.Tl_0 + 9 * self.Tl_1 - 9 / 2 * self.Tl_2 + self.Tl_3
        self.D5 = self.Tl_0

        self.A6 = -9 / 2 * self.Tr_0 + 27 / 2 * self.Tr_1 - 27 / 2 * self.Tr_2 + 9 / 2 * self.Tr_3
        self.B6 = 9 * self.Tr_0 - 45 / 2 * self.Tr_1 + 18 * self.Tr_2 - 9 / 2 * self.Tr_3
        self.C6 = -11 / 2 * self.Tr_0 + 9 * self.Tr_1 - 9 / 2 * self.Tr_2 + self.Tr_3
        self.D6 = self.Tr_0

        self.Y_up = [round(self.A1 * i ** 3 + self.B1 * i ** 2 + self.C1 * i + self.D1, 4) for i in m]
        self.T_m = [round(self.A2 * i ** 3 + self.B2 * i ** 2 + self.C2 * i + self.D2, 4) for i in m]
        self.T_l = [round(self.A5 * i ** 3 + self.B5 * i ** 2 + self.C5 * i + self.D5, 4) for i in m]
        self.T_r = [round(self.A6 * i ** 3 + self.B6 * i ** 2 + self.C6 * i + self.D6, 4) for i in m]

        self.Anglel_0, self.Anglel_1, self.Anglel_2, self.Anglel_3 = self.Angler_0_initial, self.Anglel_1_initial, self.Anglel_2_initial, self.Anglel_3_initial
        self.Angler_0, self.Angler_1, self.Angler_2, self.Angler_3 = self.Angler_0_initial, self.Angler_1_initial, self.Angler_2_initial, self.Angler_3_initial

        if self.Type == 0:
            self.R_l = [round(-self.X_l[i] ** 2 / (2 * (self.Y_l[i] - (self.Y_up[i] - self.T_m[i] / 2))), 4) for i in
                        range(self.Layer_num)]
            self.R_r = [round(-self.X_r[i] ** 2 / (2 * (self.Y_r[i] - (self.Y_up[i] - self.T_m[i] / 2))), 4) for i in
                        range(self.Layer_num)]
            self.Angle_l = [round(math.atan(self.X_l[i] / self.R_l[i]) * 180 / math.pi, 4) for i in
                            range(self.Layer_num)]
            self.Angle_r = [round(math.atan(self.X_r[i] / self.R_r[i]) * 180 / math.pi, 4) for i in
                            range(self.Layer_num)]
        else:
            self.Anglel_0 = self.Angler_0_initial * para_list[16]
            self.Anglel_1 = self.Anglel_1_initial * para_list[17]
            self.Anglel_2 = self.Anglel_2_initial * para_list[18]
            self.Anglel_3 = self.Anglel_3_initial * para_list[19]
            self.Angler_0 = self.Angler_0_initial * para_list[20]
            self.Angler_1 = self.Angler_1_initial * para_list[21]
            self.Angler_2 = self.Angler_2_initial * para_list[22]
            self.Angler_3 = self.Angler_3_initial * para_list[23]

            self.A9 = -9 / 2 * self.Anglel_0 + 27 / 2 * self.Anglel_1 - 27 / 2 * self.Anglel_2 + 9 / 2 * self.Anglel_3
            self.B9 = 9 * self.Anglel_0 - 45 / 2 * self.Anglel_1 + 18 * self.Anglel_2 - 9 / 2 * self.Anglel_3
            self.C9 = -11 / 2 * self.Anglel_0 + 9 * self.Anglel_1 - 9 / 2 * self.Anglel_2 + self.Anglel_3
            self.D9 = self.Anglel_0
            self.A10 = -9 / 2 * self.Angler_0 + 27 / 2 * self.Angler_1 - 27 / 2 * self.Angler_2 + 9 / 2 * self.Angler_3
            self.B10 = 9 * self.Angler_0 - 45 / 2 * self.Angler_1 + 18 * self.Angler_2 - 9 / 2 * self.Angler_3
            self.C10 = -11 / 2 * self.Angler_0 + 9 * self.Angler_1 - 9 / 2 * self.Angler_2 + self.Angler_3
            self.D10 = self.Angler_0

            self.Angle_l = [round(self.A9 * i ** 3 + self.B9 * i ** 2 + self.C9 * i + self.D9, 4) for i in m]
            self.Angle_r = [round(self.A10 * i ** 3 + self.B10 * i ** 2 + self.C10 * i + self.D10, 4) for i in m]
            if self.Type == 1:
                self.R_l = [round(
                    -self.X_l[i] ** 2 / (self.Y_l[i] - (self.Y_up[i] - self.T_m[i] / 2)) - self.X_l[i] / math.tan(
                        self.Angle_l[i] * math.pi / 180), 4) for i in range(self.Layer_num)]
                self.R_r = [round(
                    -self.X_r[i] ** 2 / (self.Y_r[i] - (self.Y_up[i] - self.T_m[i] / 2)) - self.X_r[i] / math.tan(
                        self.Angle_r[i] * math.pi / 180), 4) for i in range(self.Layer_num)]
                self.b_l = [-2 * self.R_l[i] for i in range(self.Layer_num)]
                self.a_l = [
                    1 / math.tan(self.Angle_l[i] * math.pi / 180) ** 2 - self.b_l[i] ** 2 / (4 * self.X_l[i] ** 2) for i
                    in range(self.Layer_num)]
                self.b_r = [-2 * self.R_r[i] for i in range(self.Layer_num)]
                self.a_r = [
                    1 / math.tan(self.Angle_r[i] * math.pi / 180) ** 2 - self.b_r[i] ** 2 / (4 * self.X_r[i] ** 2) for i
                    in range(self.Layer_num)]
            elif self.Type == 2:
                for i in range(self.Layer_num):
                    a, b, theta = para_type2(self.X_l[i], self.Y_l[i] - (self.Y_up[i] - self.T_m[i] / 2),
                                             self.Angle_l[i])
                    self.a_l.append(a);
                    self.b_l.append(b);
                    self.theta_l.append(theta)
                    a, b, theta = para_type2(self.X_r[i], self.Y_r[i] - (self.Y_up[i] - self.T_m[i] / 2),
                                             self.Angle_r[i])
                    self.a_r.append(a);
                    self.b_r.append(b);
                    self.theta_r.append(theta)
                self.R_l = [(1 + self.b_l[i] ** 2) ** 0.5 * self.a_l[i] for i in range(self.Layer_num)]
                self.R_r = [(1 + self.b_r[i] ** 2) ** 0.5 * self.a_r[i] for i in range(self.Layer_num)]
        Rlm = np.polyfit(m, self.R_l, 3)
        self.A3, self.B3, self.C3, self.D3 = Rlm
        Rrm = np.polyfit(m, self.R_r, 3)
        self.A4, self.B4, self.C4, self.D4 = Rrm
        self.delta = (self.B1 - self.B2) ** 2 - 3 * (self.A1 - self.A2) * (self.C1 - self.C2)
        if self.delta < 0:
            self.m0__ = -100
        else:
            self.m0__ = ((-(self.B1 - self.B2) - math.sqrt(
                (self.B1 - self.B2) ** 2 - 3 * (self.A1 - self.A2) * (self.C1 - self.C2))) * (
                                 self.Z_top - self.Z_bot)) / (
                                3 * (self.A1 - self.A2))

    def IsConstraintAvailable(self):
        # 1. 初始化统一的不满足率跟踪字典（拆分了平顺性）
        self.violation_flags = {
            '中心角异常(<=0或>=90)': False,
            '不满足驻点(m0__)': False,
            '左右两拱端相交': False,
            '拱圈线两端间距过小': False,
            '曲率半径小于0': False,
            '出现倒角': False,
            '上游面倒悬度>0.3': False,
            '下游面倒悬度>0.3': False,
            '角度平顺性不满足': False,
            '厚度平顺性不满足': False,
            '厚度不满足': False,
            '上游面不满足保凸性': False,
            '下游面不满足保凸性': False
        }

        self.Angle_l_, self.Angle_r_ = np.array(self.Angle_l), np.array(self.Angle_r)

        if np.any((self.Angle_l_ <= 0) | (self.Angle_l_ >= 90) | (self.Angle_r_ <= 0) | (self.Angle_r_ >= 90)):
            self.violation_flags['中心角异常(<=0或>=90)'] = True
            return False

        self.Available = []
        self.R_l_up, self.R_l_down, self.R_r_up, self.R_r_down = [], [], [], []
        self.h = [i - self.Z_bot for i in self.Gong_h]
        self.m = [i / (self.Z_top - self.Z_bot) for i in self.h]

        for i in range(self.Layer_num):
            self.x_mid_l = [j / 10 * self.X_l[i] for j in range(11)]
            self.x_mid_r = [-j / 10 * self.X_r[i] for j in range(11)]
            if self.Type == 0:
                self.angle_l = [math.atan(self.x_mid_l[j] / self.R_l[i]) for j in range(11)]
                self.angle_r = [math.atan(-self.x_mid_r[j] / self.R_r[i]) for j in range(11)]
            elif self.Type == 1:
                self.angle_l = [math.atan(
                    2 * self.x_mid_l[j] / math.sqrt(4 * self.a_l[i] * self.x_mid_l[j] ** 2 + self.b_l[i] ** 2)) for j in
                    range(11)]
                self.angle_r = [math.atan(
                    -2 * self.x_mid_r[j] / math.sqrt(4 * self.a_r[i] * self.x_mid_r[j] ** 2 + self.b_r[i] ** 2)) for j
                    in range(11)]
            elif self.Type == 2:
                self.angle_l = [j * self.Angle_l[i] / 10 / 180 * math.pi for j in range(11)]
                self.angle_r = [j * self.Angle_r[i] / 10 / 180 * math.pi for j in range(11)]
                self.x_mid_l = [(self.a_l[i] * np.exp(self.b_l[i] * self.angle_l[j]) * np.sin(
                    self.theta_l[i] + self.angle_l[j]) - self.a_l[i] * np.sin(self.theta_l[i])) for j in range(11)]
                self.x_mid_r = [-(self.a_r[i] * np.exp(self.b_r[i] * self.angle_r[j]) * np.sin(
                    self.theta_r[i] + self.angle_r[j]) - self.a_r[i] * np.sin(self.theta_r[i])) for j in range(11)]

            self.Sa_l = parabola_arc_length(self.X_l[i], 2 * self.R_l[i])
            self.S_l = [parabola_arc_length(self.x_mid_l[j], 2 * self.R_l[i]) for j in range(11)]

            self.T_d_l = [self.T_m[i] + (self.T_l[i] - self.T_m[i]) * (self.S_l[j] / self.Sa_l) ** 2 for j in range(11)]
            self.x_up_l = [self.x_mid_l[j] + self.T_d_l[j] / 2 * math.sin(self.angle_l[j]) for j in range(11)]
            self.x_down_l = [self.x_mid_l[j] - self.T_d_l[j] / 2 * math.sin(self.angle_l[j]) for j in range(11)]

            self.Sa_r = parabola_arc_length(self.X_r[i], 2 * self.R_r[i])
            self.S_r = [parabola_arc_length(self.x_mid_r[j], 2 * self.R_r[i]) for j in range(11)]

            self.T_d_r = [self.T_m[i] + (self.T_r[i] - self.T_m[i]) * (self.S_r[j] / self.Sa_r) ** 2 for j in range(11)]
            self.x_up_r = [self.x_mid_r[j] - self.T_d_r[j] / 2 * math.sin(self.angle_r[j]) for j in range(11)]
            self.x_down_r = [self.x_mid_r[j] + self.T_d_r[j] / 2 * math.sin(self.angle_r[j]) for j in range(11)]

            if self.Type == 0:
                self.y_mid_l = [self.Y_up[i] - self.T_m[i] / 2 + self.x_mid_l[j] ** 2 / (-2 * self.R_l[i]) for j in
                                range(11)]
                self.y_mid_r = [self.Y_up[i] - self.T_m[i] / 2 + self.x_mid_r[j] ** 2 / (-2 * self.R_r[i]) for j in
                                range(11)]
            elif self.Type == 1:
                self.y_mid_l = [self.Y_up[i] - self.T_m[i] / 2 + (
                        -self.b_l[i] - math.sqrt(self.b_l[i] ** 2 + 4 * self.a_l[i] * self.x_mid_l[j] ** 2)) / 2 /
                                self.a_l[i] for j in range(11)]
                self.y_mid_r = [self.Y_up[i] - self.T_m[i] / 2 + (
                        -self.b_r[i] - math.sqrt(self.b_r[i] ** 2 + 4 * self.a_r[i] * self.x_mid_r[j] ** 2)) / 2 /
                                self.a_r[i] for j in range(11)]
            elif self.Type == 2:
                self.y_mid_l = [
                    self.Y_up[i] - self.T_m[i] / 2 - self.a_l[i] * math.cos(self.theta_l[i]) + self.a_l[i] * math.exp(
                        self.b_l[i] * self.angle_l[j]) * math.cos(self.theta_l[i] + self.angle_l[j]) for j in range(11)]
                self.y_mid_r = [
                    self.Y_up[i] - self.T_m[i] / 2 - self.a_r[i] * math.cos(self.theta_r[i]) + self.a_r[i] * math.exp(
                        self.b_r[i] * self.angle_r[j]) * math.cos(self.theta_r[i] + self.angle_r[j]) for j in range(11)]

            self.y_up_l = [self.y_mid_l[j] + self.T_d_l[j] / 2 * math.cos(self.angle_l[j]) for j in range(11)]
            self.y_down_l = [self.y_mid_l[j] - self.T_d_l[j] / 2 * math.cos(self.angle_l[j]) for j in range(11)]
            self.y_up_r = [self.y_mid_r[j] + self.T_d_r[j] / 2 * math.cos(self.angle_r[j]) for j in range(11)]
            self.y_down_r = [self.y_mid_r[j] - self.T_d_r[j] / 2 * math.cos(self.angle_r[j]) for j in range(11)]

            if self.Type == 0:
                self.R_l_up.append(round(-self.x_up_l[-1] ** 2 / (2 * (self.y_up_l[-1] - self.Y_up[i])), 4))
                self.R_l_down.append(
                    round(-self.x_down_l[-1] ** 2 / (2 * (self.y_down_l[-1] - (self.Y_up[i] - self.T_m[i]))), 4))
                self.R_r_up.append(round(-self.x_up_r[-1] ** 2 / (2 * (self.y_up_r[-1] - self.Y_up[i])), 4))
                self.R_r_down.append(
                    round(-self.x_down_r[-1] ** 2 / (2 * (self.y_down_r[-1] - (self.Y_up[i] - self.T_m[i]))), 4))
            elif self.Type == 1:
                self.R_l_up.append(round(
                    -self.x_up_l[-1] ** 2 / (self.y_up_l[-1] - self.Y_up[i]) - self.x_up_l[-1] / math.tan(
                        self.Angle_l[i] * math.pi / 180), 4))
                self.R_r_up.append(round(
                    -self.x_up_r[-1] ** 2 / (self.y_up_r[-1] - self.Y_up[i]) - self.x_up_r[-1] / math.tan(
                        self.Angle_r[i] * math.pi / 180), 4))
                self.R_l_down.append(round(
                    -self.x_up_l[-1] ** 2 / (self.y_down_l[-1] - (self.Y_up[i] - self.T_m[i])) - self.x_up_l[
                        -1] / math.tan(self.Angle_l[i] * math.pi / 180), 4))
                self.R_r_down.append(round(
                    -self.x_up_r[-1] ** 2 / (self.y_down_r[-1] - (self.Y_up[i] - self.T_m[i])) - self.x_up_r[
                        -1] / math.tan(self.Angle_r[i] * math.pi / 180), 4))
            elif self.Type == 2:
                self.R_l_up += [1]
                self.R_r_up += [1]
                self.R_l_down += [1]
                self.R_r_down += [1]

            constraint_R_l_up = [self.R_l_up[i] > 0]
            constraint_R_r_up = [self.R_r_up[i] > 0]
            constraint_R_l_down = [self.R_l_down[i] > 0]
            constraint_R_r_down = [self.R_r_down[i] > 0]

            constraint_x_l_d = [self.x_down_l[j + 1] > self.x_down_l[j] for j in range(10)]
            constraint_x_l_u = [self.x_up_l[j + 1] > self.x_up_l[j] for j in range(10)]
            constraint_x_r_d = [self.x_down_r[j + 1] < self.x_down_r[j] for j in range(10)]
            constraint_x_r_u = [self.x_up_r[j + 1] < self.x_up_r[j] for j in range(10)]

            constraint_downface = [self.x_down_l[j] - self.x_down_r[j] >= 0 for j in range(11)]
            constraint_downface_special = [self.x_down_l[-1] - self.x_down_r[-1] >= 10]
            constraint_GongGup = [6 * self.A1 * self.m[i] + 2 * self.B1 < 0]
            constraint_GongGdown = [6 * (self.A1 - self.A2) * self.m[i] + 2 * (self.B1 - self.B2) < 0]

            if i == self.Layer_num - 1:
                self.overhang_up = abs(self.C1 / (self.Z_top - self.Z_bot))
                self.overhang_down = abs((self.C1 - self.C2) / (self.Z_top - self.Z_bot))
                constraint_overhang_up = self.overhang_up <= 0.35
                constraint_overhang_down = self.overhang_down <= 0.35

                # 【修改】分别计算角度平顺和厚度平顺
                constraint_smooth_anglel = (self.Angle_l_ini - 6 < self.Angle_l[-1]) and (
                            self.Angle_l[-1] < self.Angle_l_ini + 9)
                constraint_smooth_angler = (self.Angle_r_ini - 5 < self.Angle_r[-1]) and (
                            self.Angle_r[-1] < self.Angle_r_ini + 8)
                is_smooth_angle = constraint_smooth_anglel and constraint_smooth_angler

                constraint_smooth_Tm = all(
                    self.T_m[i + 1] > self.T_m[i] - 4 * ((i + 1) / 10) ** 0.4 for i in range(len(self.T_m) - 1))
                constraint_smooth_Tl = all(
                    self.T_l[i + 1] > self.T_l[i] - 5 * ((i + 1) / 10) ** 0.4 for i in range(len(self.T_l) - 1))
                constraint_smooth_Tr = all(
                    self.T_r[i + 1] > self.T_r[i] - 5 * ((i + 1) / 10) ** 0.4 for i in range(len(self.T_r) - 1))
                constraint_T = all([18 >= self.T_m[0] >= 9.95 and 18 >= self.T_l[0] >= 10 and 18 >= self.T_r[0] >= 10])
                is_smooth_thickness = constraint_smooth_Tm and constraint_smooth_Tl and constraint_smooth_Tr
                is_thickness =constraint_T

                if not (self.Z_top - self.Z_bot >= self.m0__ >= 0):
                    self.violation_flags['不满足驻点(m0__)'] = True
                    self.Available.append(False)

            if False in constraint_downface:
                self.violation_flags['左右两拱端相交'] = True
                self.Available.append(False)

            if False in constraint_downface_special:
                self.violation_flags['拱圈线两端间距过小'] = True
                self.Available.append(False)

            if False in constraint_R_r_up or False in constraint_R_l_up or False in constraint_R_r_down or False in constraint_R_l_down:
                self.violation_flags['曲率半径小于0'] = True
                self.Available.append(False)

            if False in constraint_x_l_d or False in constraint_x_r_d or False in constraint_x_l_u or False in constraint_x_r_u:
                self.violation_flags['出现倒角'] = True
                self.Available.append(False)

            if i == self.Layer_num - 1:
                if not constraint_overhang_up:
                    self.violation_flags['上游面倒悬度>0.3'] = True
                    self.Available.append(False)
                if not constraint_overhang_down:
                    self.violation_flags['下游面倒悬度>0.3'] = True
                    self.Available.append(False)

                # 【修改】独立判断角度和厚度的平顺性
                if not is_smooth_angle:
                    self.violation_flags['角度平顺性不满足'] = True
                    self.Available.append(False)
                if not is_smooth_thickness:
                    self.violation_flags['厚度平顺性不满足'] = True
                    self.Available.append(False)
                if not is_thickness:
                    self.violation_flags['厚度不满足'] = True
                    self.Available.append(False)
            # The optimizer applies the convexity checks to
            # the preceding layers; the final-layer branch is reserved for
            # overhang, stationarity, angle and thickness checks.
            if i < self.Layer_num - 1:
                if False in constraint_GongGup:
                    self.violation_flags['上游面不满足保凸性'] = True
                    self.Available.append(False)

                if False in constraint_GongGdown:
                    self.violation_flags['下游面不满足保凸性'] = True
                    self.Available.append(False)

        if False in self.Available:
            return False
        else:
            return True


def main():
    # 1. 设定体形参数的上下界
    LB = [0.87, 0.87, 0.935, 0.97, 0.973452, 0.950634, 0.883, 1.00, 0.883, 0.857, 0.857, 0.883, 0.942085, 0.857,
          0.857, 0.883]
    UB = [1.13, 1.13, 1.065, 1.08, 1.06, 1.062991, 1.01548, 1.0795, 1.117, 1.143, 1.143, 1.117, 1.117, 1.0858, 1.143,
          1.117]

    # 2. LHS 生成 10000 个测试样本
    num_samples = 10000
    print(f"\n正在进行 LHS 拉丁超立方采样 ({num_samples}个点)...")
    sampler = qmc.LatinHypercube(d=16, seed=2026)
    sample_lhd = sampler.random(n=num_samples)
    samples = qmc.scale(sample_lhd, LB, UB)

    model = DamParameter(Type=0)
    model.InitialParameter()

    # 3. 初始化全局违反统计字典（更新了键名）
    total_valid = 0
    violation_counts = {
        '中心角异常(<=0或>=90)': 0,
        '不满足驻点(m0__)': 0,
        '左右两拱端相交': 0,
        '拱圈线两端间距过小': 0,
        '曲率半径小于0': 0,
        '出现倒角': 0,
        '上游面倒悬度>0.3': 0,
        '下游面倒悬度>0.3': 0,
        '角度平顺性不满足': 0,  # 【修改】记录角度违背
        '厚度平顺性不满足': 0,  # 【修改】记录厚度违背
        '厚度不满足': 0,
        '上游面不满足保凸性': 0,
        '下游面不满足保凸性': 0
    }

    print("开始评估 10000 个样本的各项约束，这可能需要几十秒...")
    start_time = time.time()

    # 4. 遍历样本评估
    for idx in range(num_samples):
        model.ParameterUpdate(samples[idx])
        is_valid = model.IsConstraintAvailable()

        if is_valid:
            total_valid += 1

        # 统计触发的具体不满足情况
        for key, has_violated in model.violation_flags.items():
            if has_violated:
                violation_counts[key] += 1

        if (idx + 1) % 2000 == 0:
            print(f" -> 已处理 {idx + 1} / {num_samples} 个样本...")

    end_time = time.time()

    # 5. 输出详尽结果
    print("\n" + "=" * 50)
    print(f"评估完成！总耗时: {end_time - start_time:.2f} 秒")
    print(f"总抽样数: {num_samples} 个")
    print(f"完全满足所有约束的成功样本数: {total_valid}")
    print(f"总体可行率 (Feasibility Rate): {total_valid / num_samples * 100:.3f}%\n")
    print("-" * 50)
    print("各项未注释约束的独立不满足率 (Violation Rate):")
    for k, v in violation_counts.items():
        rate = v / num_samples * 100
        print(f"  - {k:<20}: {rate:>6.2f}%  (共 {v} 次)")
    print("=" * 50)


if __name__ == "__main__":
    main()
