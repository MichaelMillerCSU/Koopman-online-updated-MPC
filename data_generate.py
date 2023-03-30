import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as scio
from scipy import optimize
import math
from sympy import *
from scipy import linalg

class generate():
    def __init__(self, n_step, n_traj):
        self.N = n_step
        self.N_Traj = n_traj

    def duffing_generate(self):
        m = 1
        n = 2
        h = 0.05
        N = self.N
        N_Traj = self.N_Traj
        ## Rk4-Solve
        f = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])
        k1 = lambda t, x, u: np.array(f(t, x, u))
        k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
        k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
        k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
        f_update = lambda t, x, u: np.array(
            x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))


        u0 = 4.0 * np.random.rand(N, N_Traj) - 2.0
        # u0 = np.zeros((0, N_Traj))
        # u1 = 4.0 * np.random.rand(1, N_Traj) - 2.0
        # for i in range (0, N):
        #    u0 = np.concatenate((u0, u1), axis=0)
        # compareU = scio.loadmat('compareU.mat')
        # UU = compareU['U']
        # u0[0: 100][0] = UU
        x0 = 4.0 * np.random.rand(n, N_Traj) - 2.0
        # x0[0][0] = -2.0
        # x0[1][0] = -1.5
        # x0[:, 0] = np.array([0.0, 0.07])
        x = x0;
        X = np.zeros((n, 0))
        Y = np.zeros((n, 0))
        U = np.zeros((m, 0))
        for i in range(0, N):
            nowu = np.array(u0[i, :])
            x_next = np.array(f_update(0, x, nowu))
            x_next = np.reshape(x_next, (n, N_Traj))
            X = np.concatenate((X, x), axis=1)
            Y = np.concatenate((Y, x_next), axis=1)
            nowu = nowu.reshape((m, N_Traj))
            U = np.concatenate((U, nowu), axis=1)
            x = x_next

        X_temp = np.zeros((n, 0))
        Y_temp = np.zeros((n, 0))
        U_temp = np.zeros((m, 0))

        for i in range(0, N_Traj):
            for j in range(0, N):
                x = X[:, i + j * N_Traj]
                x = np.reshape(x, (n, 1))
                y = Y[:, i + j * N_Traj]
                y = np.reshape(y, (n, 1))
                u = U[:, i + j * N_Traj]
                u = np.reshape(u, (m, 1))

                X_temp = np.concatenate((X_temp, x), axis=1)
                Y_temp = np.concatenate((Y_temp, y), axis=1)
                U_temp = np.concatenate((U_temp, u), axis=1)

        X = X_temp
        Y = Y_temp
        U = U_temp
        return X, Y, U

    def vanderpol_generate(self):
        # 龙格库塔
        m = 1
        n = 2
        h = 0.05
        N = self.N
        N_Traj = self.N_Traj
        '''
        def f(t, x, u):
            x1 = x[0, :];
            x2 = x[1, :];
            return 2 * x2, -0.8 * x1 + 2 * x2 - 10 * x1 ** 2 * x2 + u
        '''

        f = lambda t, x, u: np.array(
            [2.0 * x[1, :], 2.0 * x[1, :] - 10.0 * x[0, :] ** 2.0 * x[1, :] - 0.8 * x[0, :] + u])
        k1 = lambda t, x, u: np.array(f(t, x, u))
        k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
        k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
        k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
        f_update = lambda t, x, u: np.array(
            x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))
        u0 = 4.0 * np.random.rand(N, N_Traj) - 2.0
        # u0 = np.zeros((0, N_Traj))
        # u1 = 4.0 * np.random.rand(1, N_Traj) - 2.0
        # for i in range (0, N):
        #    u0 = np.concatenate((u0, u1), axis=0)
        # compareU = scio.loadmat('compareU.mat')
        # UU = compareU['U']
        # u0[0 : 100][0] = UU
        x0 = 4.0 * np.random.rand(n, N_Traj) - 2.0
        # x0[0][0] = 1.0
        # x0[1][0] = 1.0
        # x0[:, 0] = np.array([0.0, 0.07])
        x = x0;
        X = np.zeros((n, 0))
        Y = np.zeros((n, 0))
        U = np.zeros((m, 0))

        for i in range(0, N):
            nowu = np.array(u0[i, :])
            x_next = np.array(f_update(0, x, nowu))
            x_next = np.reshape(x_next, (n, N_Traj))
            X = np.concatenate((X, x), axis=1)
            Y = np.concatenate((Y, x_next), axis=1)
            nowu = nowu.reshape((m, N_Traj))
            U = np.concatenate((U, nowu), axis=1)
            x = x_next

        # prepare for the data
        X_temp = np.zeros((n, 0))
        Y_temp = np.zeros((n, 0))
        U_temp = np.zeros((m, 0))

        for i in range(0, N_Traj):
            for j in range(0, N):
                x = X[:, i + j * N_Traj]
                x = np.reshape(x, (n, 1))
                y = Y[:, i + j * N_Traj]
                y = np.reshape(y, (n, 1))
                u = U[:, i + j * N_Traj]
                u = np.reshape(u, (m, 1))

                X_temp = np.concatenate((X_temp, x), axis=1)
                Y_temp = np.concatenate((Y_temp, y), axis=1)
                U_temp = np.concatenate((U_temp, u), axis=1)

        X = X_temp
        Y = Y_temp
        U = U_temp

        return X, Y, U









