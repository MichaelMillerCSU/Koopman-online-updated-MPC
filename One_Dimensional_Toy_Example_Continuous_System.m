clc
clear
close all
f_u = @(t, x, u) ([ 0.2 * x^2 - 0.3 * x^3 + 0.4 * x+ u]);
deltaT = 0.05;
%Runge-Kutta 4
% k1 = @(t,x,u) (  f_u(t,x,u) );
% k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
% k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
% k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
% f = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u,deltaT) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u,deltaT) ( f_u(t,x + k2(t,x,u,deltaT)*deltaT/2,u) );
k4 = @(t,x,u,deltaT) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
f_ud = @(t,x,u,deltaT) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u,deltaT) + 2*k3(t,x,u,deltaT) + k4(t,x,u,deltaT)  )   );

p = 10000 * rand;
% rng(2.600013010007292e+03)

% rng(2.935483432543247e+03)
% rng(3.796642527049246e+03)
% rng(2.266510471025863e+03)
rng(4.403453914021329e+03)
phi = @(x) [x; Encoder_One_Dimensional_System(x)];

Nlift = size(phi(0), 1);

Ntraj = 2000;
Nsim = 1;
PhiX_p = [];
PhiX_f = [];
U = [];
X_p = [];
Y_p = [];
Sample_Set = [];


for i = 1 : Ntraj
    u = 2 * rand - 1;

    x = 2 * rand - 1;
    deltaT = 0.05;
    Sample_Set = [Sample_Set deltaT];
    PhiX_p = [PhiX_p phi(x)]; 
    X_p = [X_p x];
    x_plus = f_ud(0, x, u,deltaT);
    Y_p = [Y_p x_plus];
    PhiX_f = [PhiX_f phi(x_plus)]; 
    x = x_plus;
    U = [U u];
end

PhiX_p_Tilde = [PhiX_p; U];

K_ext = PhiX_f * PhiX_p_Tilde' * pinv(PhiX_p_Tilde * PhiX_p_Tilde')

A = K_ext (1 : Nlift, 1 : Nlift)
B = K_ext (:, Nlift + 1 : end)
% C = [eye(1) zeros(1, Nlift - 1)];
C = X_p * PhiX_p' * pinv(PhiX_p * PhiX_p')

deltaT = 0.05;
k1 = @(t,x,u) (  f_u(t,x,u) );
k2 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT/2,u) );
k3 = @(t,x,u) ( f_u(t,x + k2(t,x,u)*deltaT/2,u) );
k4 = @(t,x,u) ( f_u(t,x + k1(t,x,u)*deltaT,u) );
f = @(t,x,u) ( x + (deltaT/6) * ( k1(t,x,u) + 2*k2(t,x,u) + 2*k3(t,x,u) + k4(t,x,u)  )   );
% Testing
Error = [];
N_init_test = 5;
X0 = linspace(-1, 1, N_init_test);
for j = 1 : N_init_test
        x0 = X0(j);
        N_Test = 20;
    
    X_Test = [x0];
    U_Test = [];
    x = x0;
    
    for i = 1 : N_Test
        u = rand;
        U_Test = [U_Test u];
        x_plus = f(0, x, u);
        X_Test = [X_Test x_plus];
        x = x_plus;
    end
    
    phix = phi(x0);
    Phi_Test = [phix];
    X_Recover = [C * phix];
    for i = 1 : N_Test
        u = U_Test(i);
        phix_plus = A * phix + B * u;
        x_plus_recover = C * phix_plus;
        X_Recover = [X_Recover x_plus_recover];
        phix = phix_plus;
    %     if mod(i, 2) == 0 
    %         phix = phi(x_plus_recover);
    %     end
    end
    
    Time = 0 : N_Test;
%     plot(Time(1), X_Test(1), 'Marker','*', 'Color','r')
%     plot(Time(1), X_Recover(1), 'Marker','+', 'Color','b')
    plot(Time, X_Test, 'LineWidth', 2.0, 'Color','r');
    hold on 
    plot(Time, X_Recover, 'LineWidth', 2.0, 'Color','b');
    legend("Original system", "Koopman operator nominal model")
    xlabel("$Steps$", 'Interpreter','latex')
    ylabel("$State$", 'Interpreter','latex')
    Error = [Error ; abs(X_Test - X_Recover)];
end
% figure
% plot(Sample_Set, 'Marker','*')

% figure
% plot(Time, sum(Error, 1)  / size(Error, 1), 'LineWidth', 2.0, 'Color','r');
% axis([0 20 -0.1 0.1])
% legend("Average MAE")
% xlabel("$Steps$", 'Interpreter','latex')
% ylabel("$State$", 'Interpreter','latex')




% 
% N_Test = 20;
% 
% X_Test = [x0];
% U_Test = [];
% x = x0;
% 
% for i = 1 : N_Test
%     u = rand;
%     U_Test = [U_Test u];
%     x_plus = f(0, x, u);
%     X_Test = [X_Test x_plus];
%     x = x_plus;
% end
% 
% phix = phi(x0);
% Phi_Test = [phix];
% X_Recover = [C * phix];
% for i = 1 : N_Test
%     u = U_Test(i);
%     phix_plus = A * phix + B * u;
%     x_plus_recover = C * phix_plus;
%     X_Recover = [X_Recover x_plus_recover];
%     phix = phix_plus;
% %     if mod(i, 2) == 0 
% %         phix = phi(x_plus_recover);
% %     end
% end
% 
% Time = 0 : N_Test;
% plot(Time, X_Test, 'LineWidth', 2.0);
% hold on 
% plot(Time, X_Recover, 'LineWidth', 2.0);
% legend("Original System", "Koopman operator model")
% xlabel("$Steps$", 'Interpreter','latex')
% ylabel("$State$", 'Interpreter','latex')
% 



