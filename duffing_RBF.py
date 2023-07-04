import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as scio
from scipy import optimize
import math
from scipy import linalg
import data_generate
from sympy import symbols, Matrix
from lmi_sdp import LMI_PD, LMI_NSD
import time
from sklearn.cluster import KMeans
from scipy.integrate import solve_ivp
from scipy.interpolate import griddata
from sklearn.metrics.pairwise import euclidean_distances


def rbf(X, cx):
    distx = euclidean_distances(X.reshape(-1,1).transpose(), cx)
    phix = np.power(distx, 2) * np.log(distx + 1e-4)
    return phix.transpose()


# data = scio.loadmat('DeepLearning_KoopmanControl_Approach3_Vanderpol_13.mat')
np.random.seed(101)


Nlift = 8
n = 2
m = 1

criterion = nn.MSELoss(reduction='sum')

N = 100
N_Traj = 100

data_generator = data_generate.generate(N, N_Traj)
X,Y,U = data_generator.duffing_generate();
# X,Y,U = van_der_pol_solve(N, N_Traj)
# X,Y,U = duffing_RK_solve(N, N_Traj)

kmeansx = KMeans(Nlift)
kmeansx.fit(X.transpose())
cx = kmeansx.cluster_centers_

print(X.shape)
print(Y.shape)
print(U.shape)

plt.figure()
#plt.subplot(311)
# plt.title('Trajectories of Duffing Oscillator used for training')

# for i in range (3):
plt.scatter(X[0, :], X[1, :], s=3)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.show()


# 数据预处理
inputs_x = X
inputs_y = Y
inputs_u = U

batch = N
batch_size = N_Traj

batch = N
batch_size = N_Traj


PHIX = np.zeros((Nlift, 0))
PHIY = np.zeros((Nlift, 0))

for items in range(0, N * N_Traj):
    g_x = rbf(inputs_x[:, items], cx)
    g_x = np.reshape(g_x, (len(g_x), 1))

    g_x_prime = rbf(inputs_y[:, items], cx)
    g_x_prime = np.reshape(g_x_prime, (len(g_x_prime), 1))

    PHIX = np.concatenate((PHIX, g_x), axis = 1)
    PHIY = np.concatenate((PHIY, g_x_prime), axis = 1)

    # nowu = inputs_u[items, :]
    # nowu = torch.reshape(nowu, (m, m))
    persent = (items + 1) / (N * N_Traj) * 100
    print("\rCalculating dictionary of X and Y ： {:3}% ".format(persent), end="")


PHIY_U = np.concatenate((PHIX, inputs_u), axis = 0)
K_hat = PHIY @ np.linalg.pinv(PHIY_U)
A_hat = K_hat[0: Nlift, 0: Nlift]
B_hat = K_hat[0: Nlift, Nlift].reshape(-1, 1)
#A = eta * A_hat + (1 - eta) * A_init
#B = eta * B_hat + (1 - eta) * B_init
A = A_hat
B = B_hat
#PHIX = PHIX.detach().numpy()
C = X @ np.linalg.pinv(PHIX)







# Dynamic system
h = 0.05
fv = lambda t, x, u: np.array([2.0 * x[1, :], 2.0 * x[1, :] - 10.0 * x[0, :] ** 2.0 * x[1, :] - 0.8 * x[0, :] + u])
# -0.5 * x2 + x1 - x1**3
# fv = lambda t, x, u: np.array([2.0 * x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])

fd = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])
f = fd
k1 = lambda t, x, u: np.array(f(t, x, u))
k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
f_update = lambda t, x, u: np.array(x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))




np.random.seed(101)
X,Y,U = data_generator.duffing_generate();

inputs_x = torch.tensor(X)
inputs_y = torch.tensor(Y)
inputs_u = torch.tensor(U)

# MPC Close-Loop Control
def costFunction(uSequence, r , AB, C, x0, pastu, Np, Nc, d):
    logY = np.zeros((2, 0))
    uSequence = np.array(uSequence)
    i = 0
    n_x = x0.shape[0]

    for u in uSequence:
        u=np.reshape(u,(1,1))
        x_u = np.concatenate([x0, u])
        x_u = np.reshape(x_u, (n_x + 1, 1))
        x0 = AB @ x_u + d
        y = C @ x0

        y = np.reshape(y, (2, 1))
        y = y - np.array(r[:, i]).reshape(2, 1)
        if i == 0 :
            y = y ;
        logY = np.concatenate((logY, y), axis=1)
        i = i + 1


    for i in range (Nc, Np):
        u = np.array(uSequence[-1]).reshape((1,1))
        x_u = np.concatenate([x0, u])
        x_u = np.reshape(x_u, (n_x + 1, 1))
        x0 = AB @ x_u + d
        y = C @ x0
        y = np.reshape(y, (2, 1))
        y = y - np.array(r[:, i]).reshape(2, 1)

        logY = np.concatenate((logY, y), axis=1)
    #print('\nerror:', y)
    #    logY+=[y[0][1]]
   # logY = np.array(logY)
    #    print(logY-r)
    #P = np.identity(8)
    delta_u = uSequence[1:] - uSequence[:-1]
    x0 = np.array(x0)
    x0 = np.reshape(x0, (len(x0), 1))
    x = x0.transpose()
    cost = 100 * np.sum(np.square(logY)) + 0.0001 * np.sum(np.square(uSequence)) #+ 0.1 * np.sum(np.square(delta_u)+uSequence[0] - pastu)
    return cost
#
def solve_DARE(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 500
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T @ X @ A - A.T @ X @ B @ np.linalg.pinv(R + B.T @ X @ B) @ B.T @ X @ A + Q
        if (abs(Xn - X)).max()  < eps:
            X = Xn
            break
        X = Xn

    return Xn

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_DARE(A, B, Q, R)

    # compute the LQR gain
    K = np.linalg.pinv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K

maxStep = 10000
tspan = h * np.arange(0, maxStep)

MPCHorizon=10
ControlHorizon = 10
pastRes=np.zeros((ControlHorizon));
pastRes_loc=np.zeros((ControlHorizon));
bounds=[(-2,2) for i in range(0,ControlHorizon)]
nx, nu = B.shape

# This is the testing problem for MPC controller. Now its more silent :)

logUloc = np.zeros((1, 0))
logU = np.zeros((1, 0))
logYR = np.zeros((maxStep, 0))
logX = np.zeros((2, 0))
logXloc = np.zeros((2, 0))
logXlift = np.zeros((8, 0))
logXLOClift = np.zeros((8, 0))
logR = np.zeros((2, 0))
init = np.array([-2.0, -2.0])
# init = np.array([-0.0, -0.0])
x0 = init

#x0 = torch.zeros((1, 2))
x_origin = np.array(init).reshape((2,1))
x_loc = np.array(init).reshape((2,1))

ctrb = np.zeros((Nlift, 0))

for i in range (A.shape[0]):
    ctrb_part = np.linalg.matrix_power(A, i) @ B
    ctrb = np.concatenate([ctrb, ctrb_part], axis=1)

Rank = np.linalg.matrix_rank(ctrb)

Q = 10*np.identity(Nlift)
R = 0.01
K_gain = dlqr(A,B,Q,R)
K_gain = np.reshape(K_gain, (1, 8))
P = solve_DARE(A, B, Q, R)
print(P)


# LTV-Parameters
# Y_EX = np.zeros((Nlift, 0))
# X_EX = np.zeros((Nlift, 0))
# U_EX = np.zeros((m, 0))
# XU_EX = np.zeros((m + Nlift, 0))
Y_EX = PHIY
X_EX = PHIX
U_EX = inputs_u
XU_EX = PHIY_U
T_EX = []
A_error = []
B_error = []
C_error = []
# t1 = time.time()
u = 0.0
# Closed loop simulation
for i in range(0,maxStep):
    #
    # if i < maxStep / 2.0:
    #     # r1=[ 0.7*np.array([[np.sin(j/(20+0.01*j))]])+.7 for j in range(i,i+MPCHorizon)]
    #     r1=[1 * np.array([[np.sin(0.01 * j)]]) for j in range(i, i + MPCHorizon)]
    # else:
    #     r1 = [np.array(1 * np.power(-1, math.ceil(i / 200)) + 0.01 * np.random.rand(1,1) - 0.01 / 2.0) for j in range(i, i + MPCHorizon)]
    #     # r1 = 2 * np.ones((1, MPCHorizon))
    # r1 = [1 * np.array([[np.sin(0.007 * j)]]) for j in range(i, i + MPCHorizon)]
    # r1 = [np.array(1 * np.power(-1, math.ceil(i / 3000))) for j in range(i, i + MPCHorizon)]
    r1 = [np.array(1.0) for j in range(i, i + MPCHorizon)]

    # r1 = [np.array(0 * np.power(1, math.ceil(i / 1000))) for j in range(i, i + MPCHorizon)]
    # r1 = 3 * np.ones((1, MPCHorizon))
    # r1 = [0.7 * np.array([[np.sin(j / (20 + 0.01 * j))]])+.7 - ek  for j in range(i, i + MPCHorizon)]
    #r1 = [1 * np.array([[np.sin(0.01 * j)]]) for j in range(i, i + MPCHorizon)]
    #r1 = [0.5 * np.array([[np.cos(0.007 * j )]]) + 1.2 * np.array([[np.sin(0.002 * j )]]) for j in range(i, i + MPCHorizon)]
    r1 = np.array(r1).reshape((1, MPCHorizon))
    # r2 = [0.1 * np.array([[np.cos(0.002 * j )]]) for j in range(i, i + MPCHorizon)]
    r2 = np.zeros((1, MPCHorizon))
    r2 = np.array(r2).reshape((1, MPCHorizon))
    r = np.concatenate([r1, r2], axis=0)
    print("\nsteps : ", i)
    print("\nr : ",r[:, 0])
    # logR = np.concatenate([logR, np.array(r[:, 0]).reshape(2, 1)], axis=1)
    #r = np.ones((1, 8))
    x0 = rbf(x0, cx)
    xlift = x0
    xlift = np.reshape(xlift, (8, 1))
    logXlift = np.concatenate((logXlift, xlift), axis=1)

    x_init = x0
    AB = np.concatenate([A, B], axis=1)

    # Koopman MPC
    if i % 1 == 0:
        lamdaCostFunction=lambda x: costFunction(x,r,AB,C,x_init,u, MPCHorizon, ControlHorizon, np.zeros((8, 1)))
        result=optimize.minimize(lamdaCostFunction,pastRes,bounds=bounds)
        u=np.array(result.x[0]).reshape((1,1))
    print("\nu : ", u)
    # u = 0.0 * np.array(result.x[0]).reshape((1,1))
    yk = C @ (A @ x_init + B @ u);



    x_origin = np.array(f_update(0, x_origin, u))
    x_origin = np.array(x_origin)
    x_origin = x_origin.astype(float)
    x_origin = np.reshape(x_origin,(2, 1))
    ek = yk[0, 0] - x_origin[0, 0]
    x0 = x_origin
    logU = np.concatenate((logU, u), axis=1)
    logX = np.concatenate((logX, x_origin), axis=1)
    print("\nx_origin : ", x_origin)


    if i > 100:
        f = lambda t, x, u: np.array([x[1, :], -10.0 * 0.5 * x[1, :] + 2.0 * x[0, :] - 0.5 * x[0, :] ** 3.0 + u])

    pass
print('\n')
# # scio.savemat('DeepLearning_KoopmanControl_Approach3_vanderpol_11.mat', {'A': A,'B': B,'PHIX' : PHIX.detach().numpy(), 'X':X, 'Y':Y , 'x0': x_init, 'logU': logU})

# t1_end = time.time()

Aloc_d = A
Bloc_d = B
Cloc_d = C
Ap = A;
Bp = B
Cp = C
f = fd
# t2 = time.time()
x0 = init

t_sum = 0

for i in range(0,maxStep):
    #
    # if i < maxStep / 2.0:
    #     # r1=[ 0.7*np.array([[np.sin(j/(20+0.01*j))]])+.7 for j in range(i,i+MPCHorizon)]
    #     r1=[1 * np.array([[np.sin(0.01 * j)]]) for j in range(i, i + MPCHorizon)]
    # else:
    #     r1 = [np.array(1 * np.power(-1, math.ceil(i / 200)) + 0.01 * np.random.rand(1,1) - 0.01 / 2.0) for j in range(i, i + MPCHorizon)]
    #     # r1 = 2 * np.ones((1, MPCHorizon))
    # r1 = [1 * np.array([[np.sin(0.007 * j)]]) for j in range(i, i + MPCHorizon)]
    # r1 = [np.array(1 * np.power(-1, math.ceil(i / 5000))) for j in range(i, i + MPCHorizon)]
    # r1 = [np.array(0 * np.power(1, math.ceil(i / 1000))) for j in range(i, i + MPCHorizon)]
    r1 = [np.array(1.0) for j in range(i, i + MPCHorizon)]
    # r1 = [0.7 * np.array([[np.sin(j / (20 + 0.01 * j))]])+.7 - ek  for j in range(i, i + MPCHorizon)]
    #r1 = [1 * np.array([[np.sin(0.01 * j)]]) for j in range(i, i + MPCHorizon)]
    #r1 = [0.5 * np.array([[np.cos(0.007 * j )]]) + 1.2 * np.array([[np.sin(0.002 * j )]]) for j in range(i, i + MPCHorizon)]
    r1 = np.array(r1).reshape((1, MPCHorizon))
    # r2 = [0.1 * np.array([[np.cos(0.002 * j )]]) for j in range(i, i + MPCHorizon)]
    r2 = np.zeros((1, MPCHorizon))
    r2 = np.array(r2).reshape((1, MPCHorizon))
    r = np.concatenate([r1, r2], axis=0)
    logR = np.concatenate([logR, np.array(r[:, 0]).reshape(2, 1)], axis=1)
    print("\nsteps : ", i)
    print("\nr : ",r[:, 0])
    #r = np.ones((1, 8))
    x0 = rbf(x0, cx)
    xlift = x0
    xlift = np.reshape(xlift, (Nlift, 1))
    logXLOClift = np.concatenate((logXLOClift, xlift), axis=1)

    x_init = x0

    ABloc_d = np.concatenate((Aloc_d, Bloc_d), axis=1)
    lamdaCostFunction = lambda x: costFunction(x, r, ABloc_d, Cloc_d, x_init, 0.0, MPCHorizon, ControlHorizon,
                                               np.zeros((8, 1)))
    result_loc=optimize.minimize(lamdaCostFunction,pastRes_loc,bounds=bounds)

    u_loc=np.array(result_loc.x[0]).reshape((1,1))

    print("\nuloc : ", u_loc)


    x_loc = np.array(f_update(0, x_loc, u_loc))
    x_loc = np.array(x_loc)
    x_loc = x_loc.astype(float)
    x_loc = np.reshape(x_loc, (2, 1))
    x0 = x_loc
    logUloc = np.concatenate((logUloc, u_loc), axis=1)
    logXloc = np.concatenate((logXloc, x_loc), axis=1)

    print("\nx_loc : ", x_loc)

    ylift = x_loc
    ylift = rbf(ylift, cx)
    ylift = np.reshape(ylift, (Nlift, 1))


    # Update LTV-Model

    X_EX = np.concatenate((X_EX, xlift), axis = 1)
    Y_EX = np.concatenate((Y_EX, ylift), axis = 1)
    U_EX = np.concatenate((U_EX, u_loc), axis = 1)
    XU_EX = np.concatenate((X_EX, U_EX), axis = 0)
    X = np.concatenate((X, x_loc), axis = 1)
    # K_hat = PHIY @ torch.pinverse(PHIY_U)
    # A_hat = K_hat[0: Nlift, 0: Nlift]
    # B_hat = K_hat[0: Nlift, Nlift]
    # B_hat = torch.reshape(B_hat, (len(B), m))
    xlift_u = np.concatenate((xlift, u_loc), axis=0)
    # K_A = (M * K_A + q * ylift @ xlift_u.transpose()) / (M + q)
    # K_G = (M * K_G + q * xlift_u @ xlift_u.transpose()) / (M + q)


    # if i <= 100:
    #     K_A = Y_EX @ XU_EX.transpose() / (M + i + 1)
    #     K_G = XU_EX @ XU_EX.transpose() / (M + i + 1)
    #     inv_K_G = np.linalg.pinv(K_G)
    #     K_ext = K_A @ np.linalg.pinv(K_G)
    # else :
    #     # K_A = ((i + 1) * K_A + 1.0 * ylift @ xlift_u.transpose()) / ((i + 1) + 1.0)
    #     K_A = K_A + ylift @ xlift_u.transpose()
    #     inv_K_G = inv_K_G - (inv_K_G @ xlift_u @ xlift_u.transpose() @ inv_K_G) / (1 + xlift_u.transpose() @ inv_K_G @ xlift_u)
    #     # K_G = ((i + 1) * K_G + q * xlift_u @ xlift_u.transpose()) / ((i + 1) + q)
    #     K_ext = K_A @ inv_K_G
    t1 = time.time()

    # 存储法在线更新
    K_A = Y_EX @ XU_EX.transpose()
    K_G = XU_EX @ XU_EX.transpose()
    inv_K_G = np.linalg.pinv(K_G)
    K_ext = K_A @ inv_K_G
    C_prev = X  @ np.linalg.pinv(X_EX)

    # # 带自定义的满秩矩阵作为初值
    # if i == 0 :
    #     K_A = np.zeros((Nlift, Nlift + m))
    #     inv_K_G = 0.0001 * np.identity(Nlift + m)
    #     inv_K_G = np.linalg.pinv(inv_K_G)
    #     K_A = K_A + ylift @ xlift_u.transpose()
    #     inv_K_G = inv_K_G - (inv_K_G @ xlift_u @ xlift_u.transpose() @ inv_K_G) / (
    #                 1 + xlift_u.transpose() @ inv_K_G @ xlift_u)
    # else:
    #     K_A = K_A + ylift @ xlift_u.transpose()
    #     inv_K_G = inv_K_G - (inv_K_G @ xlift_u @ xlift_u.transpose() @ inv_K_G) / (
    #             1 + xlift_u.transpose() @ inv_K_G @ xlift_u)
    # K_ext = K_A @ inv_K_G
    #
    #
    #
    # ## 带自定义的值作为初值
    # if i == 0 :
    #     bar_X = np.zeros((n, Nlift))
    #     bar_X = bar_X + x_loc @ xlift.transpose()
    #     bar_Q = 100.0 * np.identity(Nlift)
    #     bar_Q = bar_Q - (bar_Q @ xlift @ xlift.transpose() @ bar_Q) / (
    #                 1 + xlift.transpose() @ bar_Q @ xlift)
    # else:
    #     bar_X = bar_X + x_loc @ xlift.transpose()
    #     bar_Q = bar_Q - (bar_Q @ xlift @ xlift.transpose() @ bar_Q) / (
    #             1 + xlift.transpose() @ bar_Q @ xlift)
    # C_prev = bar_X @ bar_Q





    t1_end = time.time()
    t_sum = t_sum + t1_end - t1




    A_prev = K_ext[0: Nlift, 0: Nlift]
    B_prev = K_ext[0: Nlift, Nlift]
    B_prev = np.reshape(B_prev, (len(B_prev), m))


    T_EX.append(i * h)
    # print("A error: ", np.linalg.norm(Aloc_d - A_prev, ord=2))
    # print("B error: ", np.linalg.norm(Bloc_d - B_prev, ord=2))
    # print("C error: ", np.linalg.norm(Cloc_d - C_prev, ord=2))
    # A_error.append(np.linalg.norm((Aloc_d - A_prev), ord=2))
    # B_error.append(np.linalg.norm(Bloc_d - B_prev, ord=2))
    # C_error.append(np.linalg.norm(Cloc_d - C_prev, ord=2))

    if i % 1 == 0:
        Ap = Aloc_d
        Bp = Bloc_d
        Cp = Cloc_d
        Aloc_d = A_prev
        Bloc_d = B_prev
        Cloc_d = C_prev
    print("A error: ", np.linalg.norm(Aloc_d - Ap, ord=2))
    print("B error: ", np.linalg.norm(Bloc_d - Bp, ord=2))
    print("C error: ", np.linalg.norm(Cloc_d - Cp, ord=2))
    A_error.append(np.linalg.norm((Aloc_d - Ap), ord=2))
    B_error.append(np.linalg.norm(Bloc_d - Bp, ord=2))
    C_error.append(np.linalg.norm(Cloc_d - Cp, ord=2))
    if i > 100:
        f = lambda t, x, u: np.array([x[1, :], -10.0 * 0.5 * x[1, :] + 2.0 * x[0, :] - 0.5 * x[0, :] ** 3.0 + u])

    '''
    plt.clf()
    plt.plot(np.linspace(0.0, (i + 1) * h, i + 1),  logX[0, :], label="y_real", linewidth=3.0)
    plt.legend(loc="lower right")
    # plt.plot(tspan, logR[0, :], label = "Ref = $0.5cos(0.007t) + 1.2sin(0.002t)$",linewidth=2.0, linestyle='--')#sin(j/(20+0.01*j))
    # plt.plot(tspan, logR[0, :], label = "Ref = $sin(t/(20+0.01*t))$",linewidth=2.0, linestyle='--')
    plt.plot(np.linspace(0.0, (i + 1) * h, i + 1), logR[0, :], label="y_ref", linewidth=2.0, linestyle='--')
    # plt.plot(tspan, logR[0, :], label = "Ref = squared wave, T = 50s",linewidth=2.0, linestyle='--')
    plt.legend(loc="lower right")
    # plt.plot(tspan, logR[1, :], label = "Ref_2 ",linewidth=2.0, linestyle='--')

    plt.grid()
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.title('MPC Tracking trajectory')

    plt.pause(0.001)
    '''
    pass
print('\n')

scio.savemat('DuffingPlotrealtime.mat', {'tspan': tspan,'logXLOClift': logXLOClift,'logXloc' : logXloc, 'logR':logR, 'logX':logX , 'T_EX': T_EX, 'A_error': A_error, 'A_error':A_error, 'B_error':B_error, 'C_error':C_error, 'tspan_pred':tspan_pred, 'X':X, 'test_Y':test_Y, 'marker_T':marker_T, 'marker_originX':marker_originX, 'test_X':test_X, 'decoder_X':decoder_X, 'marker_X':marker_X, 'Uplot':Uplot
})
# scio.savemat('DuffingPlot.mat', {'tspan': tspan, 'logR': logR, 'logUloc': logUloc, 'logU': logU, 'logXLOClift': logXLOClift,'logXloc' : logXloc, 'logR':logR, 'logX':logX , 'T_EX': T_EX, 'A_error': A_error, 'A_error':A_error, 'B_error':B_error, 'C_error':C_error, 'tspan_pred':tspan_pred, 'X':X, 'test_Y':test_Y, 'marker_T':marker_T, 'marker_originX':marker_originX, 'test_X':test_X, 'decoder_X':decoder_X, 'marker_X':marker_X, 'Uplot':Uplot
# })
print("\nKoopman算子更新总共时间：",t_sum)

plt.figure()
for i in range (0, 8):
    plt.plot(tspan, logXLOClift[i, :], label = "lifted trajectory %d"%(i + 1))
    plt.legend(loc="lower right")
plt.grid()
# plt.title('Lifted trajectories')
plt.xlabel('$t/s$')
plt.ylabel('Lifted $x$ ')
plt.ylim(-3, 3)

plt.figure()
plt.tight_layout()

plt.plot(tspan, logX[0, :], label = "output trajectory $y$ without online update",linewidth=3.0)
plt.legend(loc="lower right")
#plt.plot(tspan, logX[1, :], label = "x_2 trajectory ",linewidth=3.0)
# plt.legend(loc="lower right")
#plt.plot(tspan, logR[0, :], label = "Ref = $0.5cos(0.007t) + 1.2sin(0.002t)$",linewidth=2.0, linestyle='--')#sin(j/(20+0.01*j))
#plt.plot(tspan, logR[0, :], label = "Ref = $sin(t/(20+0.01*t))$",linewidth=2.0, linestyle='--')
#plt.plot(tspan, logR[0, :], label = "Ref = $0.7sin(t/(20+0.01t))+0.7$",linewidth=2.0, linestyle='--')

plt.plot(tspan, logXloc[0, :], label = "output trajectory $\hat{y}$ with online update",linewidth=3.0)
plt.plot(tspan, logR[0, :], label = "reference",linewidth=3.0, linestyle='--')
pltR = plt.legend(loc="lower right")
#plt.plot(tspan, logR[1, :], label = "Ref_2 ",linewidth=2.0, linestyle='--')
# pltR = plt.legend(loc="lower right")
plt.grid()
plt.ylim(-2.5, 2.2)
plt.xlabel('$t/s$')
plt.ylabel('$y$')
plt.tight_layout()

# plt.title('MPC Tracking trajectory')

#plt.legend([[pltX1], [pltX2],[pltR1], [pltR2]], ['x_1 trajectory', 'x_2 trajectory', 'Ref_1 = 1*sin(0.001x)', 'Ref_2 = 0'], loc='lower right')

plt.figure()

pltPhase, = plt.plot(logX[0, :], logX[1, :], label = "x_1 and x_2 phase plots ")
plt.legend(loc="lower right")
#plt.legend([[pltPhase]], ['x_1 and x_2 Phase plots'], loc='lower right')
plt.grid()
plt.tight_layout()


plt.figure()
pltU, = plt.plot(tspan, logU.ravel(), label = "input $u$ without update ")
pltU, = plt.plot(tspan, logUloc.ravel(), label = "input $u$ with update ")
pltU, = plt.plot(tspan, 2.0 * np.ones(maxStep), linestyle = '--', color = 'black')
pltU, = plt.plot(tspan, -2.0 * np.ones(maxStep), linestyle = '--', color = 'black', label = "input bounds")


plt.legend(loc="lower right")
plt.grid()
plt.ylim(-2.3, 2.3)
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$u$')
plt.tight_layout()
# plt.title('Control inputs')

plt.figure()


# plt.plot(T_EX, A_error / np.linalg.norm(A_error, ord=np.inf), label = "A_error  ")
plt.plot(T_EX, A_error)

# plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$(||A_{(k+1)} - A_k||_{2}^{2}) / A_k||_{2}^{2}$')
# plt.xlim(0, 10)
# plt.ylim(0, 0.1)
# plt.title('Matrix A error')
plt.tight_layout()


plt.figure()
plt.tight_layout()

# pltMatrixErrorB, = plt.plot(T_EX, B_error / np.linalg.norm(B_error, ord=np.inf), label = "B_error ")
pltMatrixErrorB, = plt.plot(T_EX, B_error)

# plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$(||B_{(k+1)} - B_k||_{2}^{2}) / B_k||_{2}^{2}$')
# plt.xlim(0, 10)
# plt.ylim(0, 0.01)
# plt.title('Matrix B error')
plt.tight_layout()

plt.figure()
plt.tight_layout()

# pltMatrixErrorB, = plt.plot(T_EX, B_error / np.linalg.norm(B_error, ord=np.inf), label = "B_error ")
pltMatrixErrorC, = plt.plot(T_EX, C_error)

# plt.legend(loc="lower right")
plt.grid()
#plt.legend([[pltU]], ['Control inputs'], loc='lower right')
plt.xlabel('$t/s$')
plt.ylabel('$(||C_{(k+1)} - C_k||_{2}^{2}) / C_k||_{2}^{2}$')
# plt.xlim(0, 10)
# plt.ylim(0, 0.01)
# plt.title('Matrix B error')
plt.tight_layout()
# #locally linear plots
# plt.figure()
#
# plt.plot(tspan, logXloc[0, :], label = "y = $x_1$ trajectory ",linewidth=3.0)
# plt.legend(loc="lower right")
# #plt.plot(tspan, logX[1, :], label = "x_2 trajectory ",linewidth=3.0)
# plt.legend(loc="lower right")
# #plt.plot(tspan, logR[0, :], label = "Ref = $0.5cos(0.007t) + 1.2sin(0.002t)$",linewidth=2.0, linestyle='--')#sin(j/(20+0.01*j))
# #plt.plot(tspan, logR[0, :], label = "Ref = $sin(t/(20+0.01*t))$",linewidth=2.0, linestyle='--')
# #plt.plot(tspan, logR[0, :], label = "Ref = $0.7sin(t/(20+0.01t))+0.7$",linewidth=2.0, linestyle='--')
#
# plt.plot(tspan, logR[0, :], label = "Reference",linewidth=2.0, linestyle='--')
# pltR = plt.legend(loc="lower right")
# #plt.plot(tspan, logR[1, :], label = "Ref_2 ",linewidth=2.0, linestyle='--')
# pltR = plt.legend(loc="lower right")
# plt.grid()
# plt.xlabel('$t/s$')
# plt.ylabel('$y$')
# plt.title('MPC Tracking trajectory (locally linear)')
#
# #plt.legend([[pltX1], [pltX2],[pltR1], [pltR2]], ['x_1 trajectory', 'x_2 trajectory', 'Ref_1 = 1*sin(0.001x)', 'Ref_2 = 0'], loc='lower right')
#
# plt.figure()
# pltPhase, = plt.plot(logXloc[0, :], logXloc[1, :], label = "x_1 and x_2 Phase plots ")
# plt.legend(loc="lower right")
# #plt.legend([[pltPhase]], ['x_1 and x_2 Phase plots'], loc='lower right')
# plt.grid()
#
# plt.figure()
# pltU, = plt.plot(tspan, logUloc.ravel(), label = "Control inputs ")
# plt.legend(loc="lower right")
# plt.grid()
# #plt.legend([[pltU]], ['Control inputs'], loc='lower right')
# plt.xlabel('$t/s$')
# plt.ylabel('$u$')
# plt.title('Control inputs (locally linear)')
#


scio.savemat('RBF_Thinplate.mat', {"X_Collection_NO": logX, "X_Collection": logXloc, "U_Collection" : logU})






plt.show()

