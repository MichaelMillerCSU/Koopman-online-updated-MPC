clc
clear
close all
rng(55)
%% *************************** Dynamics ***********************************
% fd = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])

% f_u =  @(t,x,u)([ 2*x(2,:) ; 2.0*x(2, :) - 10.0*x(1, :).^2.*x(2, :) - 0.8*x(1, :) + u] );
f_ud =  @(t,x,u)([ x(1,:) - 0.5 * sqrt(x(1,:)) + 0.4 .* u ; ...
                   x(2,:) + 0.2 * sqrt(x(1,:)) - 0.3 * sqrt(x(2,:)) ] );


n = 2;
m = 1; % number of control inputs


% ************************** Discretization ******************************

% deltaT = 0.01;
% %Runge-Kutta 4
% k1 = @(t,x,u) (  f_u(t,x,u) );
% k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
% k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
% k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
% f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );



%% ************************** Collect data ********************************
tic
disp('Starting data collection')
Nsim = 100;
Ntraj = 100;

% Random forcing
Ubig = 10 * rand([Nsim Ntraj]) - 5;

% Random initial conditions
Xcurrent = ( 4 * rand(n,Ntraj) - 2 );
Xcurrent(find(Xcurrent < 0)) = 0;
X = []; Y = []; U = [];
Xlift = []; Ylift = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent,Ubig(i,:));
    Xnext(find(Xnext< 0) ) = 0;
    X = [X Xcurrent];
    Y = [Y Xnext];
    U = [U Ubig(i,:)];
    Xcurrent = Xnext;
end
fprintf('Data collection DONE, time = %1.2f s \n', toc);





%% ************************** Basis functions *****************************

basisFunction = 'rbf';
% RBF centers
Nrbf = 10;
cent = rand(n,Nrbf);
% [idx, cent_temp] = kmeans(X', Nrbf);
% cent = cent_temp';
rbf_type = 'thinplate'; 
% Lifting mapping - RBFs + the state itself
liftFun = @(xx)( [  rbf(xx,cent,rbf_type)] );
Nlift = Nrbf ;

% liftFun = @(xx) Encoder_Tank(xx);
% Nlift = 10;


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
C = M(Nlift+1:end,1:Nlift);



fprintf('Regression done, time = %1.2f s \n', toc);


nx = size(B, 1);
nu = size(B, 2);

A = [A B; zeros(nu, Nlift) eye(nu, nu)];
B = [B; eye(nu, nu)];
C = C * [eye(Nlift, Nlift) zeros(Nlift, nu)];
Cy = [0 1];


N = 20;
Q = 10 * eye(1);
R = 0.001;
Yr = [1];
Q_bar = kron(eye(N), Q);
R_bar = kron(eye(N), R);
% N = 40;
% Q_bar = kron(eye(N), Qlift);
% R_bar = kron(eye(N), Rlift);
% Q_bar(end - Nlift + 1 : end, end - Nlift + 1 : end) = P;
x0 = [0;0];
Lift_xu = [liftFun(x0); 0];
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
% Yr = kron(ones(N, 1), Yr);
for i = 1 : N - 1
%     Yr = [Yr; 1.0 + 0.7 * sin(i / 20 + 0.01 * i ) ];
    Yr = [Yr; 1.0 ];
end

umin = -8;
umax = +8;

p = size(Yr, 1);
H = (Compact_Form2)' * Q_bar * (Compact_Form2) + R_bar;
H = (H+H')/2;
U0 = 0;
A_cons = [eye(nu, nu) zeros(nu, N * nu - nu)
                  -eye(nu, nu) zeros(nu, N * nu - nu)];
b_cons = [umax * ones(nu, 1) - U0;
          -umin * ones(nu, 1) + U0];
f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
[U0_Set, feval] = quadprog(2.*H, f, A_cons, b_cons, [], [], -0.5*ones(p, 1), 0.5*ones(p, 1));
% U0 = U0_Set(1 : nu)

X_Collection = [];
U_Collection = [];
Steps = 3000;
Ref_Plot = [];


% r=[ 0.7*np.array([[np.sin(j/(20+0.01*j))]])+.7 for j in range(i,i+MPCHorizon)]
% 0.7 + 0.7 * sin(t / 20 + 0.01 t )
for i = 1 : Steps
    tic
    %% MPC Solve
    Yr = [];
    for t = i : i + N - 1
%         Yr = [Yr; 1.0 + 0.7 * sin(t / 20 + 0.01 * t ) ];
        Yr = [Yr; 1 ];
    end
    
    Ref_Plot = [Ref_Plot; Yr(1)];
    if mod(i, 1) == 0
        flag = 0;
        A_cons = [eye(nu, nu) zeros(nu, N * nu - nu)
                  -eye(nu, nu) zeros(nu, N * nu - nu)];
        b_cons = [umax * ones(nu, 1) - U0;
                  -umin * ones(nu, 1) + U0];

        f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
        [U0_Set, fval] = quadprog(2.*H, f,  A_cons, b_cons, [], [], -0.5*ones(p, 1), 0.5*ones(p, 1));
        feval = fval + (Compact_Form1 * Lift_xu)' * Q_bar * (Compact_Form1 * Lift_xu)
    end
    
    U0 = U0 + U0_Set(flag * nu + 1 : flag * nu + nu)
    if i > 100
        f_ud =  @(t,x,u)([ x(1,:) - 0.53 * sqrt(x(1,:)) + 0.3 .* u ; ...
                           x(2,:) + 0.1 * sqrt(x(1,:)) - 0.35 * sqrt(x(2,:)) ] );

%% good ----
%         f_ud =  @(t,x,u)([ x(1,:) - 0.5 * sqrt(x(1,:)) + 0.3 .* u ; ...
%                            x(2,:) + 0.1 * sqrt(x(1,:)) - 0.3 * sqrt(x(2,:)) ] );

%         f_ud =  @(t,x,u)([ x(1,:) - 0.5 * sqrt(x(1,:)) + 0.4 .* u ; ...
%                    x(2,:) + 0.2 * sqrt(x(1,:)) - 0.3 * sqrt(x(2,:)) ] );
    end

    Xlift = [Xlift [liftFun([x0])]];
    X_Collection = [X_Collection x0];

    x = x0;
    X = [X x0];
    x0 = f_ud(0, x0, U0)
    x0(find(x0 < 0)) = 0;
    U = [U U0];
    Ylift = [Ylift [liftFun([x0])]];
    y = x0;
    U_Collection = [U_Collection U0];


% % 储存法
%     W = [Ylift ; X];
%     V = [Xlift; U];
%     VVt = V*V';
%     WVt = W*V';
%     M = WVt * pinv(VVt); % Matrix [A B; C 0]
%     A = M(1:Nlift,1:Nlift);
%     B = M(1:Nlift,Nlift+1:end);
%     C = M(Nlift+1:end,1:Nlift);
%     A = [A B; zeros(nu, Nlift) eye(nu, nu)];
%     B = [B; eye(nu, nu)];
%     C = C * [eye(Nlift, Nlift) zeros(Nlift, nu)];
%     Cy = [0 1];


%% 自定义满秩
    if i == 1 
        K_A = zeros(Nlift, Nlift + nu);
%         K_A = Ylift * V';
        invK_G = 1e4 * eye(Nlift + nu);
%         invK_G = pinv(invK_G);
        invK_G = pinv(VVt);
        invK_G = invK_G - (invK_G * [liftFun([x]); U0] * [liftFun([x]); U0]'* invK_G) / (1 + [liftFun([x]); U0]' * invK_G * [liftFun([x]); U0]);
        K_A = K_A + [liftFun([y])] * [liftFun([x]); U0]';
    else
        invK_G = invK_G - (invK_G * [liftFun([x]); U0] * [liftFun([x]); U0]'* invK_G) / (1 + [liftFun([x]); U0]' * invK_G * [liftFun([x]); U0]);
        K_A = K_A + [liftFun([y])] * [liftFun([x]); U0]';
    end
    Kext = K_A * invK_G;
    A = Kext(:, 1:Nlift);
    B = Kext(:, Nlift + 1:end);

    
    if i == 1 
        bar_X = zeros(n, Nlift);
%         bar_X = X * Xlift'
%         bar_X = bar_X + x * liftFun([x])';
        bar_Q = 10000 * eye(Nlift);
%         bar_Q = pinv(Xlift * Xlift');
        bar_Q = bar_Q - (bar_Q * liftFun([x]) * liftFun([x])' * bar_Q) / (1 + liftFun([x])' * bar_Q * liftFun([x]));

    else
        bar_X = bar_X + x * liftFun([x])';
        bar_Q = bar_Q - (bar_Q * liftFun([x]) * liftFun([x])' * bar_Q) / (1 + liftFun([x])' * bar_Q * liftFun([x]));
    end
    C = bar_X * bar_Q;

    A = [A B; zeros(nu, Nlift) eye(nu, nu)];
    B = [B; eye(nu, nu)];
    C = C * [eye(Nlift, Nlift) zeros(Nlift, nu)];
    Cy = [0 1];



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
    
    Lift_xu = [liftFun(x0); U0];
end


MSE = (X_Collection(2, :) - Ref_Plot') * (X_Collection(2, :) - Ref_Plot')' / Steps


figure('Renderer', 'painters')

plot(X_Collection(2, :),'LineStyle','-','LineWidth',3.0)
hold on 
plot(Ref_Plot, 'LineStyle','--', 'Color', 'r', 'LineWidth', 2.0)


figure('Renderer', 'painters')
 
plot(U_Collection(1, :),'LineStyle','-','LineWidth',2.0)



