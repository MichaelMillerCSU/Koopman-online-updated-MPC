clc
clear
close all
r = randi(6000);
% rng(2224)
%% *************************** Dynamics ***********************************
% fd = lambda t, x, u: np.array([x[1, :], -0.5 * x[1, :] + x[0, :] - x[0, :] ** 3.0 + u])

% f_u =  @(t,x,u)([ 2*x(2,:) ; 2.0*x(2, :) - 10.0*x(1, :).^2.*x(2, :) - 0.8*x(1, :) + u] );
f_u =  @(t,x,u)([ x(2,:) ; -0.5*x(2, :) + x(1, :) - x(1, :).^3 + u] );


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
% tic
disp('Starting data collection')
Nsim = 100;
Ntraj = 100;

% Random forcing
Ubig = 4*rand([Nsim Ntraj]) - 2;

% Random initial conditions
Xcurrent = (4*rand(n,Ntraj) - 2);

X = []; Y = []; U = [];
Xlift = []; Ylift = [];
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent,Ubig(i,:));
    X = [X Xcurrent];
    Y = [Y Xnext];
    U = [U Ubig(i,:)];
    Xcurrent = Xnext;
end
% fprintf('Data collection DONE, time = %1.2f s \n', toc);

%% ************************** Basis functions *****************************

% basisFunction = 'rbf';
% % RBF centers
% Nrbf = 8;
% % cent = rand(n,Nrbf)*2 - 1;
% [idx, cent_temp] = kmeans(X', Nrbf);
% cent = cent_temp';
% rbf_type = 'thinplate'; 
% % Lifting mapping - RBFs + the state itself
% liftFun = @(xx)( [ rbf(xx,cent,rbf_type)] );
% Nlift = Nrbf;
% liftFun = @(x) [x];
% liftFun = @(x) [x; x(1)*x(2);x(1)*x(2)^2; x(1)^2*x(2)];

liftFun = @(x) [x; Encoder_Duffing(x)] - [zeros(2, 1); Encoder_Duffing(zeros(2, 1))];
% liftFun = @(x) [ Encoder_Duffing(x)];

Nlift = size(liftFun(zeros(2, 1)), 1);

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
C = M(Nlift+1:end,1:Nlift);




fprintf('Regression done, time = %1.2f s \n', toc);


nx = size(B, 1);
nu = size(B, 2);
Cy = eye(n);
%     Cy = [eye(size(Yr, 1))];
N = 10;
Q = 10* eye(2);
R = 0.01;

Q_bar = kron(eye(N), Q);
R_bar = kron(eye(N), R);
% N = 40;
% Q_bar = kron(eye(N), Qlift);
% R_bar = kron(eye(N), Rlift);
% Q_bar(end - Nlift + 1 : end, end - Nlift + 1 : end) = P;
x0 = [-1; 1];
Lift_xu = [liftFun(x0)];
% K = Lift_xu * [Lift_xu; 0]';
% A = rand(Nlift, Nlift);
% B = rand(Nlift, 1);
% C = rand(2, Nlift);

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
Yr = [1; 0];
% Yr = [];
Yr = kron(ones(N, 1), Yr);
% for i = 1 : N
%     Yr = [Yr; [sin(0.05 * i) ;0]];
% end
p = size(Yr, 1);


X_Collection = [];
U_Collection = [];
Steps = 100;
Ref_Plot = [];
time = 0;


% x_rec = x0;
x_rec = liftFun(x0);
X_Recon = [];
X_Predict = [];
T_Predict = [];
T_Init_Predict = [];
X_Init_Predict = [];



% Get Jacobian of the true dynamics (for local linearization MPC)
xx = sym('xx',[2 1]); syms u;
f_ud_sym = f_ud(0,xx,u);
u_loc = 0;
Jx = jacobian(f_ud_sym,xx);
Ju = jacobian(f_ud_sym,u);
Aloc = double(subs(Jx,[xx;u],[x0;u_loc])); % Get local linearization
Bloc = double(subs(Ju,[xx;u],[x0;u_loc]));
cloc = double(subs(f_ud_sym,[xx;u],[x0;u_loc])) - Aloc*x0 - Bloc*u_loc;

x_max = kron(ones(N , 1), [2.0; 1.0]);
x_min = kron(ones(N , 1), [-2.0; -2.0]);
AIneq = Compact_Form2;
AIneq = [AIneq; -Compact_Form2];
bIneq = [[x_max - Compact_Form1 * Lift_xu];
              [-x_min + Compact_Form1 * Lift_xu]];
H = (Compact_Form2)' * Q_bar * (Compact_Form2) + R_bar;
H = (H+H')/2;
f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
[U0_Set, feval] = quadprog(2.*H, f, [], [], [], [], -4*ones(N, 1), 4*ones(N, 1));
U0 = U0_Set(1 : nu)
epsilon_Set  = [];
epsilon_Decomposition = {};
V_Set = [];
Gamma_Set = [];
F_eval_Set = [];
A_Temp = A;
B_Temp = B;
delta_x_Set = [];
Compensator = [];
alpha_Set = [];
Compare_State = [];
Minus_Set = [];
x_predict = x0;
for i = 1 : Steps
    %% MPC Solve
    if mod(i, 1) == 0
        x_max = kron(ones(N , 1), [2.0; inf]);
        x_min = kron(ones(N , 1), [-2.0; -inf]);
        AIneq = Compact_Form2;
        AIneq = [AIneq; -Compact_Form2];
        bIneq = [[x_max - Compact_Form1 * Lift_xu];
                      [-x_min + Compact_Form1 * Lift_xu]];
        flag = 0;
        f = 2 .* (Compact_Form1 * Lift_xu)' * Q_bar * Compact_Form2 - 2 .* Yr' * Q_bar * Compact_Form2;
        [U0_Set,fval,exitflag,output,lambda]= quadprog(2.*H, f, [], [], [], [], -2*ones(N, 1), 2*ones(N, 1));
        K = -dlqr(A,B,10 * eye(Nlift, Nlift),R);
        if exitflag ~= 1 && exitflag ~= 0
            error('Infeasible!');
        end
        feval = fval + (Compact_Form1 * Lift_xu)' * Q_bar * (Compact_Form1 * Lift_xu)
        F_eval_Set = [F_eval_Set feval];
    end
    Ref_Plot = [Ref_Plot; Yr(1)];
    U0 = U0_Set(1);
%     U0 = K * Lift_xu;

%     Yr = [];
%     for k = i : i+N - 1
%         Yr = [Yr; [sin(0.01 * k) ;0]];
%     end
    if i > 100
%          f_u =  @(t,x,u)([ x(2,:) ; - 3*x(2,:) - 10*x(1,:).^2.*x(2,:)-3.0*x(1,:)  + u] );
        % f = lambda t, x, u: np.array([x[1, :], -10.0 * 0.5 * x[1, :] + 2.0 * x[0, :] - 0.5 * x[0, :] ** 3.0 + u])
        f_u =  @(t,x,u)([ x(2,:) ; - 10*x(2,:)*0.5 + 2.0 * x(1, :) - 0.5 * x(1, :).^3  + u] );
        k1 = @(t,x,u) (  f_u(t,x,u) );
        k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
        k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
        k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
        f_ud = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );
    end

    X = [X x0];
    xlift = [liftFun([x0])];
%     Xlift = [Xlift [liftFun([x0])]];
    X_Collection = [X_Collection x0];
    x = x0;
    x0 = f_ud(0, x0, U0 )
    y = x0;

    ylift = [liftFun([x0])];

    Compensator = [Compensator K * (ylift - (A * xlift + B * U0))];
%     Ylift = [Ylift [liftFun([x0])]];
    epsilon_Set = [epsilon_Set norm(x0 - C * (A * xlift + B * U0))];
    epsilon_Decomposition{i} = (ylift - (A * xlift + B * U0)) * pinv(xlift);
%     U = [U U0];
    U_Collection = [U_Collection U0 ];
    tic
    lambda = 1.0;
 %自定义满秩
    if i == 1 
        K_A = zeros(Nlift, Nlift + nu);
        invK_G = 1e4 * eye(Nlift + nu);

        K_A = Ylift * [Xlift; U]';
        invK_G = pinv([Xlift; U] * [Xlift; U]');
%        invK_G = pinv(invK_G);
%        invK_G = pinv(W);
%        invK_G = 1 * rand(Nlift + nu, Nlift + nu);

        invK_G = 1 / lambda * invK_G - 1 / lambda * (invK_G * [xlift; U0] * [xlift; U0]'* invK_G) / (lambda + [xlift; U0]' * invK_G * [xlift; U0]);
       K_A = K_A + [ylift] * [xlift; U0]';
    else
        invK_G = 1 / lambda * invK_G - 1 / lambda * (invK_G * [xlift; U0] * [xlift; U0]'* invK_G) / (lambda + [xlift; U0]' * invK_G * [xlift; U0]);
        K_A = K_A + [ylift] * [xlift; U0]';
    end
    Kext = K_A * invK_G;
    A = Kext(:, 1:Nlift);
    B = Kext(:, Nlift + 1:end);
    Q_Lift = diag([100 100 zeros(1, Nlift - n)]);

%     [K, P] = dlqr(A, B, Q_Lift, R)
% 
% %     Gamma = xlift' * P * xlift;
%     P_Set{i}  = P;
%     invariant_dif = x' * C * P* C' * x;
%     invariant_dif = fval;
%     invariant_dif = x' * x;
%     V_Set  = [V_Set invariant_dif];
% %     Gamma_Set = [Gamma_Set Gamma];

    liftFun([x0])
    x_rec  = A * x_rec + B * U0
    X_Recon = [X_Recon x_rec];
    delta_x_Set = [delta_x_Set ((A - A_Temp) * xlift + (B - B_Temp) * U0)];

%     if i == 1 
%         bar_X = zeros(n, Nlift);
% %         K_A = V;
%         bar_X = bar_X + x * xlift';
%         bar_Q = 1e2 * eye(Nlift);
% %         bar_Q = 1 * rand(Nlift, Nlift);
%         bar_Q = 1 / lambda * bar_Q - 1 / lambda * (bar_Q * xlift * xlift' * bar_Q) / (lambda + xlift' * bar_Q * xlift);
% 
%     else
%         bar_X = bar_X + x * liftFun([x])';
%         bar_Q = 1 / lambda * bar_Q - 1 / lambda * (bar_Q * xlift * xlift' * bar_Q) / (lambda + xlift' * bar_Q * xlift);
%     end
%     C = bar_X * bar_Q;
%     t2 = toc
%     time = t2 + time;



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
%             liftFun(x_predict) Q_1];s
    LMI1 = [1 liftFun(x0 - Yr(1 : 2, :))';
                liftFun(x0 - Yr(1 : 2, :)) Q_1];
%     LMI1 = [1 xlift';
%                 xlift Q_1];
    LMI2 = [(1.0) * Q_1                       (     (   A  )  * Q_1 +  B * Y_1)'    (sqrt(Q_Lift)*Q_1)'       (sqrt(R)*Y_1)'
            (  A   )  *  Q_1 + B * Y_1          Q_1                    zeros(Nlift, Nlift)       zeros(Nlift, m);
            sqrt(Q_Lift)*Q_1              zeros(Nlift, Nlift)        (gamma) * eye(Nlift, Nlift) zeros(Nlift, m);
            sqrt(R)*Y_1               zeros(m, Nlift)        zeros(m, Nlift)       (gamma) * eye(m, m)];
    

%     % Input constraints  |uk| <= 2
% 
    LMI0 = [X_1  Y_1;
            Y_1' Q_1];



    ZEROS = 0;
    Constraints = [         LMI0 >= 0;
                            LMI1 >= 0.01;
                            LMI2 >= ZEROS; 
                            Q_1 >= ZEROS];
    for j = 1 : m
        Constraints = [Constraints; 
                                X_1(j, j) <= 2^2 ] ;
    end

    Objective = gamma;

    sol  = solvesdp(Constraints)
    if sol.problem ~= 0
        error('infeasible')
    end
    Y_1 = double(Y_1)
    Q_1 = double(Q_1)
    
    K = Y_1 / Q_1
        


    Gamma = double(gamma)
    P = Gamma * inv(Q_1)



%      if (xlift'*Q_Lift * xlift - norm((2 * ylift)' * P * (ylift - (A * xlift + B * U0)))  ) < 0
%          pause;
%      end
        Minus_Set = [Minus_Set xlift'*Q_Lift * xlift - norm((2 * ylift)' * P * (ylift - (A * xlift + B * U0)))  ];


%     Q_bar = kron(eye(N), C * P * C');

    Q_bar(end - n + 1 : end, end - n + 1 : end) = C * P * C';
    P_Set{i}  = P;
    invariant_dif = liftFun(x - Yr(1:2, :))' * P * liftFun(x - Yr(1:2, :)) ;
    V_Set  = [V_Set invariant_dif];
    Gamma_Set = [Gamma_Set Gamma - (invariant_dif - (x - Yr(1:2, :))' * C * P * C' *  (x - Yr(1:2, :)))];
%     Margin = M' * P * M
    Compare_State = [Compare_State U0'*R*U0 - (A^N* N * liftFun(x0 - C* (A * xlift + B * U0)))' * P * (A^N* N * liftFun(x0 - C* (A * xlift + B * U0)))];
    
    
%     F = [A1; zeros(size(A2, 1), nx)];
%     G = [zeros(size(A1, 1), nu); A2];
%     YY = @(i) Polyhedron([K * (A + B*K)^i; -K * (A + B*K)^i;], [6; 6]);
%     
%     t = 0;
%     Y0 = YY(t);
%     while 1
%         t = t + 1;
%         Y_temp = and(Y0, YY(t));
%         if Y_temp == Y0
%             break;
%         else
%             Y0 = Y_temp;
%         end
%     end
    
%     Xf = Y0;
% %     P_Xf = plot_Set(Xf, [0.9290 0.6940 0.1250], 3);
%     O = [];
%     for o = 1 : 1e5
%         olift = 4 * rand(Nlift, 1) - 2;
% %         olift = xlift;
%         if Xf.A * olift <= Xf.b
%             O = [O olift];
%         end
%     end
%     CO = C * O;
%     scatter(CO(1, :), CO(2, :))
% %     L_Phi = norm(x0(1) - [1 0] * [C * A * xlift + C * B * U0])
%     0.01 * min(eig(P)) / norm(P, 2)
%     if L_Phi * (L_Phi + 2 * max(abs(eig(A + B * K)))) - 0.01 * min(eig(P)) / norm(P, 2) - 0.001>= 0
%         pause;
%     end

    % 储存法
%     tic
% 
%     W = [Ylift];
%     V = [Xlift; U];
%     VVt = V*V';
%     WVt = W*V';
%     M = WVt * pinv(VVt); % Matrix [A B; C 0]
%     A = M(1:Nlift,1:Nlift);
%     B = M(1:Nlift,Nlift+1:end);
%     C = (X * Xlift' ) * pinv(Xlift * Xlift');
%     t2 = toc
%     time = time + t2;

%     if i >0
% %         Aloc = double(subs(Jx,[xx;u],[x0;U0])); % Get local linearization
% %         Bloc = double(subs(Ju,[xx;u],[x0;U0]));
% %         cloc = double(subs(f_ud_sym,[xx;u],[x0;U0])) - Aloc*x0 - Bloc*U0;
%         x_predict = x;
%         T_Init_Predict = [T_Init_Predict i];
%         X_Init_Predict = [X_Init_Predict x];
%         for j = 1 : N
%             T_Predict = [T_Predict i + j];
%             x_predict = C * A * liftFun(x_predict) + C * B * U0_Set(j);
%             X_Predict = [X_Predict x_predict];
% %             x_predict = f_ud(0, x_predict, U0_Set(j));
% %             x_predict = Aloc * x0 + Bloc * U0_Set(j) ;
%             alpha_Set = [alpha_Set x_predict' * C * P * C' * x_predict];
%         end
%     end
    
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
%     Lift_xu = x_rec;
end

% MSE = (X_Collection(1, :) - Ref_Plot') * (X_Collection(1, :) - Ref_Plot')' / Steps
Steady_Error = abs(X_Collection(1, end) - Ref_Plot(1))

figure 
plot(X_Collection(1, :),'LineStyle','-','LineWidth',2.0)
hold on 
% plot(Ref_Plot(:),'LineStyle','-','LineWidth',2.0)
% plot(X_Recon(1, :),'LineStyle','-','LineWidth',2.0)
% hold on
% plot(T_Predict(1, :), X_Predict(1, :),'LineStyle','-','LineWidth',2.0)
% for i = 1 : 930
%     plot(T_Predict(i - 70 : i, :), X_Predict(i - 70 : i, :),'LineStyle','-','LineWidth',2.0)
% end
% hold on
% plot(T_Init_Predict(1, :), X_Init_Predict(1, :),'Marker','*','Color','r')
% plot(epsilon_Set(1, :),'LineStyle','-','LineWidth',2.0)
xlabel('Steps');
ylabel('$x_1$ ','interpreter','latex');

figure 
plot(X_Collection(2, :),'LineStyle','-','LineWidth',2.0)
xlabel('Steps');
ylabel('$x_2$ ','interpreter','latex');

figure 
plot(U_Collection(1, :),'LineStyle','-','LineWidth',2.0)
xlabel('Steps');
ylabel('$u$ ','interpreter','latex');

figure 
plot(epsilon_Set(1, :),'LineStyle','-','LineWidth',2.0)
xlabel('Steps');
ylabel('$\epsilon$ ','interpreter','latex');

figure
plot(V_Set,'LineWidth', 3, 'LineStyle','-')
xlabel('Steps');
ylabel('$V=\phi^TP\phi$ ','interpreter','latex');

figure
plot(diff(V_Set),'LineWidth', 3, 'LineStyle','-')
xlabel('Steps');
ylabel('$V(k+1) - V(k)$ ','interpreter','latex');


figure
for i = 1 : Steps
    t = linspace(0, 2*pi, Steps);
    z = [cos(t); sin(t)];
    R = chol( C * P_Set{i}/(Gamma_Set(i)) * C' )\z;
    h = plot(R(1,:),R(2,:),'LineWidth',3);
    hold on
%     pause(0.1)
end

% 
hh = plot(X_Collection(1, :), X_Collection(2, :),'LineStyle','-','LineWidth',3.0, 'Color', 'r')
legend(hh, 'State trajectory')
xlabel('$x_1$ ','interpreter','latex');
ylabel('$x_2$ ','interpreter','latex');



figure
plot(Gamma_Set,'LineWidth', 3, 'LineStyle','-')
xlabel('Steps');
ylabel('$\gamma_k$ ','interpreter','latex');


figure
plot(Compensator,'LineWidth', 3, 'LineStyle','-')
xlabel('Steps');
ylabel('$Comensator$ ','interpreter','latex');


figure
plot(Compare_State,'LineWidth', 3, 'LineStyle','-')
xlabel('Steps');
ylabel('$Compare_State$ ','interpreter','latex');


figure
plot(Minus_Set,'LineWidth', 3, 'LineStyle','-')
xlabel('Steps');
ylabel('$\tilde{x}^T(k)(\tilde{Q} + \tilde{F}^T\tilde{R}\tilde{F} - \Gamma_1)\tilde{x}(k)$ ','interpreter','latex');



