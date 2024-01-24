clc
clear
close all
% rng(120)
% rng(100)
%% *************************** Dynamics ***********************************
% fd = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])

f_u =  @(t,x,u)([ 2*x(2,:) ; 2.0*x(2, :) - 10.0*x(1, :).^2.*x(2, :) - 0.8*x(1, :) + u] );
% f_u =  @(t,x,u)([ x(2,:) ; -0.5*x(2, :) + x(1, :) - x(1, :).^3 + u] );


n = 2;
m = 1; % number of control inputs


%% ************************** Discretization ******************************

deltaT = 0.05;
%Runge-Kutta 4
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );



%% ************************** Collect data ********************************
tic
disp('Starting data collection')
Nsim = 100;
Ntraj = 100;

% Random forcing
Ubig = 4*rand([Nsim Ntraj]) - 2;

% Random initial conditions
Xcurrent = (rand(n,Ntraj)*4 - 2);

X = []; Y = []; U = [];
Xlift = []; Ylift = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent,Ubig(i,:));
    X = [X Xcurrent];
    Y = [Y Xnext];
    U = [U Ubig(i,:)];
    Xcurrent = Xnext;
end
fprintf('Data collection DONE, time = %1.2f s \n', toc);

%% ************************** Basis functions *****************************

basisFunction = 'rbf';
% RBF centers
Nrbf = 8;
% cent = rand(n,Nrbf)*2 - 1;
[idx, cent_temp] = kmeans(X', Nrbf);
cent = cent_temp';
rbf_type = 'thinplate'; 
% Lifting mapping - RBFs + the state itself
liftFun = @(xx)( [ rbf(xx,cent,rbf_type)] );
Nlift = Nrbf;

liftFun = @(x) [Encoder_VDP(x)] - [ Encoder_VDP(zeros(2, 1))] ;
Nlift = 8 ;

% liftFun = @(x) [x; x(1)*x(2);x(1)*x(2)^2; x(1)^2*x(2)];
% Nlift = 5;



%% ******************************* Lift ***********************************
% 
% disp('Starting LIFTING')
% tic
% Xlift = liftFun(X);
% Ylift = liftFun(Y);
% fprintf('Lifting DONE, time = %1.2f s \n', toc);

for i = 1 : size(X, 2)
    Xlift = [Xlift liftFun(X(:, i))];
    Ylift = [Ylift liftFun(Y(:, i))];
end



%% ********************** Build predictor *********************************

disp('Starting REGRESSION')
tic
W = [Ylift ; X];
V = [Xlift; U];
VVt = V*V';
WVt = W*V';
M = WVt * pinv(VVt); % Matrix [A B; C 0]
A = M(1:Nlift,1:Nlift);
B = M(1:Nlift,Nlift+1:end);
C = eye(Nlift);

fprintf('Regression done, time = %1.2f s \n', toc);


nx = size(B, 1);
nu = size(B, 2);
Cy = eye(Nlift);
%     Cy = [eye(size(Yr, 1))];
N = 10;
Q = 100 * eye(Nlift);
R = 0.0001;
Yr = liftFun([-1; 0]);
Q_bar = kron(eye(N), Q);
R_bar = kron(eye(N), R);
% N = 40;
% Q_bar = kron(eye(N), Qlift);
% R_bar = kron(eye(N), Rlift);
% Q_bar(end - Nlift + 1 : end, end - Nlift + 1 : end) = P;
x0 = [1; 1];
Lift_xu = [liftFun(x0)];
Shift_Matrix = kron([zeros(1, N); [eye(N - 1) zeros(N - 1, 1)]], eye(nu));
Compact_Form1 = [];
for i = 1 : N
    Compact_Form1 = [Compact_Form1; Cy * C * A^i];
end
Compact_Form2 = [];
for i = 1 : N
    vector_Temp = [];
    for j = 1 : N
        vector_Temp = [Cy * C * A^(j - 1)*B  vector_Temp];
    end
    Compact_Form2 = [vector_Temp * Shift_Matrix^(i - 1); Compact_Form2];
end
Yr = kron(ones(N, 1), Yr);
p = size(Yr, 1);
H = (Compact_Form2)' * Q_bar * (Compact_Form2) + R_bar;
H = (H+H')/2;
f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
[U0_Set, feval] = quadprog(2.*H, f, [], [], [], [], -6*ones(p, 1), 6*ones(p, 1));
U0 = U0_Set(1 : nu)

X_Collection = [];
U_Collection = [];
Steps = 1000;
total_t = 0;

for i = 1 : Steps
    %% MPC Solve
    if mod(i, 1) == 0
        flag = 0;
        f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
        [U0_Set, fval] = quadprog(2.*H, f, [], [], [], [], -6*ones(N, 1), 6*ones(N, 1));
        feval = fval + (Compact_Form1 * Lift_xu)' * Q_bar * (Compact_Form1 * Lift_xu)
    end
    
    U0 = U0_Set(flag * nu + 1 : flag * nu + nu);

    if i > 100
        f_u =  @(t,x,u)([ x(2,:) ; -3*x(2,:) - 10*x(1,:).^2.*x(2,:)-3.0*x(1,:)  + u] );
%         f = lambda t, x, u: np.array([x[1, :], -10.0 * 0.5 * x[1, :] + 2.0 * x[0, :] - 0.5 * x[0, :] ** 3.0 + u])
%         f_u =  @(t,x,u)([ x(2,:) ; - 10*x(2,:)*0.5 + 2.0 * x(1, :) - 0.5 * x(1, :).^3  + u] );
        k1 = @(t,x,u) (  f_u(t,x,u) );
        k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
        k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
        k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
        f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );
    end

    X = [X x0];
    Xlift = [Xlift [liftFun([x0])]];
    x = x0;
    x0 = f_ud(0, x0, U0)
    U = [U U0];
    Ylift = [Ylift [liftFun([x0])]];
    y = x0;
    U_Collection = [U_Collection U0];
    LIFTx = liftFun([x]);
    LIFTy = liftFun([y]);

tic
% 自定义满秩
    if i == 1 
        K_A = zeros(Nlift, Nlift + nu);
%         K_A = V * W';
        invK_G = 0.00001 * eye(Nlift + nu);
        invK_G = pinv(invK_G);
%         invK_G = pinv(W * W');
        invK_G = invK_G - (invK_G * [LIFTx; U0] * [LIFTx; U0]'* invK_G) / (1 + [LIFTx; U0]' * invK_G * [LIFTx; U0]);
        K_A = K_A + [LIFTy] * [LIFTx; U0]';
    else
        invK_G = invK_G - (invK_G * [LIFTx; U0] * [LIFTx; U0]'* invK_G) / (1 + [LIFTx; U0]' * invK_G * [LIFTx; U0]);
        K_A = K_A + [LIFTy] * [LIFTx; U0]';
    end
    Kext = K_A * invK_G;
    A = Kext(:, 1:Nlift);
    B = Kext(:, Nlift + 1:end);
%     Q_Lift = diag([10 10 zeros(1, Nlift - n)]);
    Q_Lift = Q;
%     if i == 1 
%         bar_X = zeros(n, Nlift);
% %         K_A = V;
%         bar_X = bar_X + x * liftFun([x])';
%         bar_Q = 100 * eye(Nlift);
%         bar_Q = bar_Q - (bar_Q * liftFun([x]) * liftFun([x])' * bar_Q) / (1 + liftFun([x])' * bar_Q * liftFun([x]));
% 
%     else
%         bar_X = bar_X + x * liftFun([x])';
%         bar_Q = bar_Q - (bar_Q * liftFun([x]) * liftFun([x])' * bar_Q) / (1 + liftFun([x])' * bar_Q * liftFun([x]));
%     end
%     C = bar_X * bar_Q;


% % 储存法
%     W = [Ylift ; X];
%     V = [Xlift; U];
%     VVt = V*V';
%     WVt = W*V';
%     M = WVt * pinv(VVt); % Matrix [A B; C 0]
% %     M = W * pinv(V);
%     A = M(1:Nlift,1:Nlift);
%     B = M(1:Nlift,Nlift+1:end);
% %     C = M(Nlift+1:end,1:Nlift);
t2 = toc;
    total_t = t2 + total_t;



    gamma = sdpvar(1, 1);
    X_1 = sdpvar(m, m);
    Q_1 = sdpvar(Nlift, Nlift);
    Y_1 = sdpvar(m, Nlift);
%     M = 0;
%     Compare_State = [Compare_State U0'*R*U0 - (A^N* N * liftFun(x0 - C* (A * xlift + B * U0)))' * P * (A^N* N * liftFun(x0 - C* (A * xlift + B * U0)))];
%     for k = 1 : N
%             M = M + A^(N - k)*delta_x_Set(:, i) ;
%     end
%     M = M + A^N* liftFun(x0 - C* (A * xlift + B * U0)) ;
%     M = diag([M(1 : n, :)./xlift(1 : n, 1); zeros(Nlift - n, 1)]);
%     maxeig = max(abs(eig(M)));
%     LMI1 = [1 liftFun(x_predict)';
%             liftFun(x_predict) Q_1];
    LMI1 = [1 (liftFun(x0 - [1; 0]))';
                liftFun(x0 - [1; 0]) Q_1];

    LMI2 = [(1.0) * Q_1                       (     (   A  )  * Q_1 +  B * Y_1)'    (sqrt(Q_Lift)*Q_1)'       (sqrt(R)*Y_1)'
            (  A   )  *  Q_1 + B * Y_1          Q_1                    zeros(Nlift, Nlift)       zeros(Nlift, m);
            sqrt(Q_Lift)*Q_1              zeros(Nlift, Nlift)        (gamma) * eye(Nlift, Nlift) zeros(Nlift, m);
            sqrt(R)*Y_1               zeros(m, Nlift)        zeros(m, Nlift)       (gamma) * eye(m, m)];
    

%     % Input constraints  |uk| <= 2
% 
    LMI0 = [X_1  Y_1;
            Y_1' Q_1];



    ZEROS = 0;
    Constraints = [
                            LMI1 >= 0;
                            LMI2 >= ZEROS; 
                            Q_1 >= ZEROS];
%     for j = 1 : m
%         Constraints = [Constraints; 
%                                 X_1(j, j) <= 6^2 ] ;
%     end

    Objective = gamma;

    sol  = solvesdp(Constraints)
    if sol.problem ~= 0
        
    end
    Y_1 = double(Y_1)
    Q_1 = double(Q_1)
    
    K = Y_1 / Q_1
        


    Gamma = double(gamma)
    P = Gamma * inv(Q_1)

    Q_bar(end - Nlift + 1 : end, end - Nlift + 1 : end) = P;

    Shift_Matrix = kron([zeros(1, N); [eye(N - 1) zeros(N - 1, 1)]], eye(nu));
    Compact_Form1 = [];
    for k = 1 : N
        Compact_Form1 = [Compact_Form1; Cy * C * A^k];
    end
    Compact_Form2 = [];
    for k = 1 : N
        vector_Temp = [];
        for j = 1 : N
            vector_Temp = [Cy * C * A^(j - 1)*B  vector_Temp];
        end
        Compact_Form2 = [vector_Temp * Shift_Matrix^(k - 1); Compact_Form2];
    end
    p = size(Yr, 1);
    H = (Compact_Form2)' * Q_bar * (Compact_Form2) + R_bar;
    H = (H+H')/2;
    f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
    Lift_xu = liftFun(x0);
    X_Collection = [X_Collection x0];
end

total_t
figure 
plot(X_Collection(1, :),'LineStyle','-','LineWidth',2.0)
figure 
plot(U_Collection(1, :),'LineStyle','-','LineWidth',2.0)








