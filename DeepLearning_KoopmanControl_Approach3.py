import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq, kmeans, whiten, kmeans2
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.metrics.pairwise import euclidean_distances
from scipy.interpolate import griddata
import scipy.io as scio


def ez_example_solve():
    def ez_example(t, x):
        x1 = x[0];
        x2 = x[1];
        return -0.1 * x1, x2 - x1 ** 2
    tspan = np.linspace(0, 1, 10)
    x0 = 10.0 * np.random.rand(1000,2) - 5.0
    sol = np.asarray([solve_ivp(ez_example, (0.0, 1.0), y0 = item, method = 'LSODA', t_eval = tspan) for item in x0])
    return sol

def duffing_solve():
    def duffing_example(t, x):
        x1 = x[0];
        x2 = x[1];
        return x2, -0.5 * x2 + x1 - x1**3
    tspan = np.linspace(0, 2.75, 11)
    x0 = np.random.uniform(-2, 2, 1000)
    y0 = np.random.uniform(-2, 2, 1000)
    data = np.array([x0, y0])
    data = data.T
    sol = np.asarray([solve_ivp(duffing_example, (0.0, 2.75), y0 = item, method = 'LSODA', t_eval = tspan) for item in data])
    return sol

def van_der_pol_solve(N, N_Traj):
    # 龙格库塔
    m = 1
    n = 2
    h = 0.01
    '''
    def f(t, x, u):
        x1 = x[0, :];
        x2 = x[1, :];
        return 2 * x2, -0.8 * x1 + 2 * x2 - 10 * x1 ** 2 * x2 + u
    '''

    f = lambda t, x, u: np.array([2.0 * x[1, :], 2.0 * x[1, :] - 10.0 * x[0, :] ** 2.0 * x[1, :] - 0.8 * x[0, :] + u])
    k1 = lambda t, x, u: np.array(f(t, x, u))
    k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
    k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
    k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
    f_update = lambda t, x, u: np.array( x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))

    u0 = 2.0 * np.random.rand(N, N_Traj) - 1.0
    x0 = 2.0 * np.random.rand(n, N_Traj) - 1.0
    #x0 = np.array([0.5, 0.5])
    x = x0;
    X = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    U = np.zeros((m, 0))

    for i in range (0, N):
        nowu = np.array(u0[i, :])
        x_next = np.array(f_update(0, x, nowu))
        x_next = np.reshape(x_next, (n, N_Traj))
        X = np.concatenate((X, x), axis=1)
        Y = np.concatenate((Y, x_next), axis=1)
        nowu = nowu.reshape((m, N_Traj))
        U = np.concatenate((U, nowu), axis=1)
        x = x_next
    return X,Y,U

def nonlinear_forApproach3_solve(N, N_Traj):
    # 龙格库塔
    m = 1
    n = 2
    h = 0.01
    '''
    def f(t, x, u):
        x1 = x[0, :];
        x2 = x[1, :];
        return 2 * x2, -0.8 * x1 + 2 * x2 - 10 * x1 ** 2 * x2 + u
    '''

    f = lambda t, x, u: np.array([-0.1 * x[0, :], -1.0 * x[1, :] + 1.0 * x[0, :] ** 4.0 - 2.0 * x[0, :] ** 2 + u])
    k1 = lambda t, x, u: np.array(f(t, x, u))
    k2 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k1(t, x, u), u))
    k3 = lambda t, x, u: np.array(f(t + h / 2.0, x + 0.5 * h * k2(t, x, u), u))
    k4 = lambda t, x, u: np.array(f(t + h, x + h * k3(t, x, u), u))
    f_update = lambda t, x, u: np.array( x + (h / 6.0) * (k1(t, x, u) + 2.0 * k2(t, x, u) + 2.0 * k3(t, x, u) + k4(t, x, u)))

    u0 = 2.0 * np.random.rand(N, N_Traj) - 1.0
    x0 = 2.0 * np.random.rand(n, N_Traj) - 1.0
    #x0 = np.array([0.5, 0.5])
    x = x0;
    X = np.zeros((n, 0))
    Y = np.zeros((n, 0))
    U = np.zeros((m, 0))

    for i in range (0, N):
        nowu = np.array(u0[i, :])
        x_next = np.array(f_update(0, x, nowu))
        x_next = np.reshape(x_next, (n, N_Traj))
        X = np.concatenate((X, x), axis=1)
        Y = np.concatenate((Y, x_next), axis=1)
        nowu = nowu.reshape((m, N_Traj))
        U = np.concatenate((U, nowu), axis=1)
        x = x_next
    return X,Y,U

def van_der_pol_solve_test(N):
    def van_der_pol_example(t, x, u):
        x1 = x[0];
        x2 = x[1];
        return 2 * x2, -0.8 * x1 + 2 * x2 - 10 * x1**2 * x2 + u

    tspan = np.linspace(0, 10.0, N)
    x0 = np.random.uniform(-2, 2)
    y0 = np.random.uniform(-2, 2)
    # 控制项需要加在 arg 参数
    sol = solve_ivp(van_der_pol_example, (0.0, 10.0), y0=(x0, y0), method='LSODA', t_eval=tspan, args=(0,))
    return sol, tspan

def van_der_pol_example_timeseries_test(N):
    solutions, tspan = van_der_pol_solve_test(N)
    X = np.zeros([2, 0])
    X_Next = np.zeros([2, 0])
    x1 = solutions.y[0, :-1]
    x2 = solutions.y[1, :-1]

    x1_next = solutions.y[0, 1:]
    x2_next = solutions.y[1, 1:]

    X_prev = np.array([x1, x2])
    X_next = np.array([x1_next, x2_next])

    return X_prev, X_next, tspan

def duffing_solve_test(N):
    def duffing_example(t, x):
        x1 = x[0];
        x2 = x[1];
        return x2, -0.5 * x2 + x1 - x1**3
    tspan = np.linspace(0, 10.0, N)
    x0 = np.random.uniform(-2, 2)
    y0 = np.random.uniform(-2, 2)

    sol = solve_ivp(duffing_example, (0.0, 10.0), y0 = (x0, y0), method = 'LSODA', t_eval = tspan)
    return sol, tspan

def nonlinear_ez_example():
    sol = ez_example_solve()
    X = np.zeros([2, 0])
    X_Next = np.zeros([2, 0])
    for solutions in sol :
        x1 = solutions.y[0, :-1]
        x2 = solutions.y[1, :-1]

        x1_next = solutions.y[0, 1:]
        x2_next = solutions.y[1, 1:]

        X_prev = np.array([x1, x2])
        X_next = np.array([x1_next, x2_next])
        X = np.concatenate((X, X_prev), axis = 1)
        X_Next = np.concatenate((X_Next, X_next), axis = 1)

    return X, X_Next

def nonlinear_duffing_example_timeseries_test(N):
    solutions, tspan = duffing_solve_test(N)
    X = np.zeros([2, 0])
    X_Next = np.zeros([2, 0])
    x1 = solutions.y[0, :-1]
    x2 = solutions.y[1, :-1]

    x1_next = solutions.y[0, 1:]
    x2_next = solutions.y[1, 1:]

    X_prev = np.array([x1, x2])
    X_next = np.array([x1_next, x2_next])

    return X_prev, X_next, tspan


def nonlinear_duffing_example():
    sol = duffing_solve()
    X = np.zeros([2, 0])
    X_Next = np.zeros([2, 0])
    for solutions in sol:
        x1 = solutions.y[0, :-1]
        x2 = solutions.y[1, :-1]

        x1_next = solutions.y[0, 1:]
        x2_next = solutions.y[1, 1:]

        X_prev = np.array([x1, x2])
        X_next = np.array([x1_next, x2_next])
        X = np.concatenate((X, X_prev), axis=1)
        X_Next = np.concatenate((X_Next, X_next), axis=1)

    return X, X_Next
def Hermite(n, x):
    if n < 0:
        return 1
    elif n == 1:
        return 2*x;
    else:
        return 2*x*Hermite(n - 1, x) - 2*(n - 1)*Hermite(n - 2, x)

def HermitePloyDist(state):
    x = state[0];
    y = state[1];
    return [Hermite(0, x)*Hermite(0, y), Hermite(1, x)*Hermite(0, y), Hermite(2, x)*Hermite(0, y), Hermite(3, x)*Hermite(0, y), Hermite(4, x)*Hermite(0, y),\
            Hermite(0, x)*Hermite(1, y), Hermite(1, x)*Hermite(1, y), Hermite(2, x)*Hermite(1, y), Hermite(3, x)*Hermite(1, y), Hermite(4, x)*Hermite(1, y),\
            Hermite(0, x)*Hermite(2, y), Hermite(1, x)*Hermite(2, y), Hermite(2, x)*Hermite(2, y), Hermite(3, x)*Hermite(2, y), Hermite(4, x)*Hermite(2, y),\
            Hermite(0, x)*Hermite(3, y), Hermite(1, x)*Hermite(3, y), Hermite(2, x)*Hermite(3, y), Hermite(3, x)*Hermite(3, y), Hermite(4, x)*Hermite(3, y),\
            Hermite(0, x)*Hermite(4, y), Hermite(1, x)*Hermite(4, y), Hermite(2, x)*Hermite(4, y), Hermite(3, x)*Hermite(4, y), Hermite(4, x)*Hermite(4, y)]


def get_phi(X,Y,M):
    phix = []
    phiy = []

    for i in range(0, M):
        x = X[i, :]
        y = Y[i, :]
        #print(x)
        phix.append(HermitePloyDist(x))
        phiy.append(HermitePloyDist(y))
    phix = np.array(phix)
    phiy = np.array(phiy)

    phix = phix.reshape(M, 25)
    phiy = phiy.reshape(M, 25)
    return phix, phiy

def get_K(phix, phiy, M):
    G = 0
    A = 0
    for i in range(0, M):
        G = phix.T @ phix
        A = phix.T @ phiy
    G = G / M
    A = A / M

    K = np.linalg.pinv(G) @ A
    return K

def snapshots(M):
    #1. Simple LTI system

    X = 10.0 * np.random.rand(M,2) - 5.0
    Y = [];
    J = [[0.9,-0.1], [0.0,0.8]]
    for x in X :
        y = J@x
        Y.append(y)
    Y = np.array(Y);
    Y.reshape(X.shape)

    return X,Y

def eigendecomposition(K):
    eigenvalues, eigenvectors = np.linalg.eig(K)

    idx = eigenvalues.real.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    left_eigenvectors = np.linalg.inv(eigenvectors)
    return eigenvalues, eigenvectors, left_eigenvectors

def get_zero(eigenvectors, eigenvalues):
    vector = []
    index = 0
    for idx in range(0, len(eigenvalues)):
        if np.abs(eigenvalues[idx].real - 1)  < 0.001 and eigenvalues[idx].imag < 1e-6:
            index = idx
            break
    return index

def plotDuffingScatter(X, eigenvectors, eigenvalues):
    print(X.shape)
    x = X[:, 0]
    y = X[:, 1]
    xi = np.arange(-2, 2, 0.001)
    yi = np.arange(-2, 2, 0.001)
    xmg, ymg = np.meshgrid(xi, yi)
    #x = eigenvalues.real
    #y = eigenvalues.imag
    plt.figure(figsize=(20, 10))
    plt.xlabel('x')
    plt.ylabel('y')
    print(len(eigenvectors))
    Za = eigenvectors[:]
    Za /= np.linalg.norm(Za.real)
    Za = Za.real
    #print(title)
    zi = griddata((x, y), Za, (xmg, ymg), method='cubic')
    im = plt.imshow(zi, cmap=plt.cm.Spectral_r, extent=(-2.0, 2.0, -2.0, 2.0))
    #plt.scatter(x, y)
    plt.colorbar(im)
    plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 8),  # compress to 3 features which can be visualized in plt
        )
        self.Decoder = nn.Sequential(
            nn.Linear(8, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            #nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.input_fc1 = nn.Linear(2, 100, bias = True)
        self.fc2 = nn.Linear(100, 100, bias = True)
        self.fc3 = nn.Linear(100, 100, bias = True)
        self.output_fc4 = nn.Linear(100, 4, bias = True)

    def forward(self, x):
        x = torch.relu(self.input_fc1(x))

        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output_fc4(x)

        x = x.squeeze(-1)

        return x

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        self.input_fc1 = nn.Linear(4, 100, bias = True)
        self.fc2 = nn.Linear(100, 100, bias = True)
        self.fc3 = nn.Linear(100, 100, bias=True)
        self.output_fc4 = nn.Linear(100, 2, bias = True)

    def forward(self, x):
        x = torch.relu(self.input_fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.output_fc4(x)

        x = x.squeeze(-1)

        return x

torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.cuda.is_available())
net = AutoEncoder()


optimizer = optim.Adam(net.parameters(), lr=0.001)
eta = 0.5

Nlift = 4

n = 2
m = 1

criterion = nn.MSELoss(reduction='sum')

N = 100
N_Traj = 100

X,Y,U = nonlinear_forApproach3_solve(N, N_Traj)

print(X.shape)
print(Y.shape)
print(U.shape)

# prepare for the data
X_temp = np.zeros((n, 0))
Y_temp = np.zeros((n, 0))
U_temp = np.zeros((m, 0))

for i in range (0, N_Traj):
    for j in range (0, N):

        x = X[:, i + j * 100]
        x = np.reshape(x, (n, 1))
        y = Y[:, i + j * 100]
        y = np.reshape(y, (n, 1))
        u = U[:, i + j * 100]
        u = np.reshape(u, (m, 1))

        X_temp = np.concatenate((X_temp, x), axis=1)
        Y_temp = np.concatenate((Y_temp, y), axis=1)
        U_temp = np.concatenate((U_temp, u), axis=1)

X = X_temp
Y = Y_temp
U = U_temp

#X = X.T
#X_prime = Y.T
#U = U.T

A_init = np.random.rand(Nlift, Nlift)
B_init = np.random.rand(Nlift, m)

A = A_init
B = B_init

# 数据预处理
inputs_x = torch.tensor(X)
inputs_y = torch.tensor(Y)
inputs_u = torch.tensor(U)
A_init = torch.tensor(A_init)
B_init = torch.tensor(B_init)


i = 0
j = 0

pred_horizon = 10
epoch = 20
batch = N
batch_size = N_Traj

PHIX = torch.tensor([])
PHIY = torch.tensor([])


alpha_1 = 1.0
alpha_2 = 50.0
alpha_3 = 50.0
alpha_4 = 1E-6
alpha_5 = 1E-6

# training
for eps in range(0, epoch):
    for i in range(0, batch):
        Loss_lin = 0
        Loss_pred = 0
        Loss_rec = 0
        Loss = 0

        pred_horizon = 6
        epoch = 20
        batch = N
        batch_size = N_Traj


        PHIX = torch.tensor([])
        PHIY = torch.tensor([])

        for items in range(0, N * N_Traj):
            g_x = net.Encoder(inputs_x[:, items])
            g_x = torch.reshape(g_x, (len(g_x), 1))
            g_x_prime = net.Encoder(inputs_y[:, items])
            g_x_prime = torch.reshape(g_x_prime, (len(g_x_prime), 1))

            PHIX = torch.cat([PHIX, g_x], dim=1)
            PHIY = torch.cat([PHIY, g_x_prime], dim=1)

            # nowu = inputs_u[items, :]
            # nowu = torch.reshape(nowu, (m, m))
            persent = (items + 1) / (N * N_Traj) * 100
            print("\rCalculating dictionary of X and Y ： {:3}% ".format(persent), end="")


        PHIY_U = torch.cat([PHIX, inputs_u], dim=0)
        K_hat = PHIY @ torch.pinverse(PHIY_U)
        A_hat = K_hat[0: Nlift, 0: Nlift]
        B_hat = K_hat[0: Nlift, Nlift]
        B_hat = torch.reshape(B_hat, (len(B), m))
        A = eta * A_hat + (1 - eta) * A_init
        B = eta * B_hat + (1 - eta) * B_init
        A_init = A.detach()
        B_init = B.detach()

        for j in range(0, batch_size):
            s_lin = 0
            s_pred = 0

            k = i * batch + j #k为当前时刻
            if N * N_Traj - k <= pred_horizon:
                break
            if batch - k <= pred_horizon:
                break
            phix = PHIX[:, k]
            x_rec = net.Decoder(phix)
            x_rec = torch.reshape(x_rec, (len(x_rec), 1))

            x_k = inputs_x[:, k]
            x_k = torch.reshape(x_k, (len(x_k), 1))
            Loss_rec = criterion(x_rec, x_k)

            for p in range(1, pred_horizon + 1):
                sum_ABu = torch.tensor(0)
                for s in range (1, p + 1):
                    tempu = inputs_u[:, k + s - 1]
                    u = torch.reshape(tempu, (m, m))
                    sum_ABu = sum_ABu + torch.matrix_power(A, s - 1) @ B @ u
                phix_k = torch.reshape(phix, (len(phix), 1))
                phix_pred = torch.matrix_power(A, p) @ phix_k + sum_ABu # A^p @ phix + \sum^s_s=1 A^(p - 1) @ B @ u
                phix_p = PHIX[:, k + p]
                phix_pred = torch.squeeze(phix_pred) # 解除warning
                #phix_p = torch.reshape(phix_p, (1 , len(phix_p)))
                #phix_pred = torch.reshape(phix_pred, (1, len(phix_pred)))
                Loss_lin = Loss_lin + criterion(phix_pred, phix_p)
                #Loss_lin = Loss_lin + s_lin

                x_p = inputs_x[:, k + p]
                #phix_pred = torch.reshape(phix_pred, (1, len(phix_pred)))
                x_rec_p = net.Decoder(phix_pred)
                Loss_pred = Loss_pred + criterion(x_p, x_rec_p)
                #Loss_pred = Loss_pred + s_pred

            Loss_lin = Loss_lin / pred_horizon
            Loss_pred = Loss_pred / pred_horizon

            weight = 0
            #Loss_lin = s_lin
            #Loss_pred = s_pred
            for param in net.parameters():
                weight = weight + torch.sum(torch.abs(param))
            if eps > 5:
                Loss = Loss + alpha_1 * Loss_rec
            else:
                Loss = Loss + alpha_1 * Loss_rec + alpha_2 * Loss_lin + alpha_3 * Loss_pred + alpha_4 * weight
            print("\rProcessing batch number ： {:d} / 100".format(j + 1), end="")

        Loss = Loss / batch_size
        print("\nLoss_rec", Loss_rec)
        print("\nLoss_lin", Loss_lin)
        print("\nLoss_pred", Loss_pred)
        print("\nLoss", Loss)
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()
        print("\nWeight updated!")

torch.save(net, 'AutoEncoder_20220324.pkl')
scio.savemat('DeepLearning_KoopmanControl_Approach3_vanderpol.mat', {'A': A,'B': B,'K_hat': K_hat, 'phix' : phix, 'X':X, 'Y':Y })






























































































































