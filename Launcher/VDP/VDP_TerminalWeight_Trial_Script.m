clc
clear
close all
Seed = 78;
rng(Seed,'twister')

% TerminalMode: 'Q' / 'LqrFixed' / 'LmiFixed' / 'LmiOnline'。

%% ************************** Parameters **********************************
n = 2;
m = 1;
Nlift = 8;

N = 10;                         % Np = Nc
Q = 100*eye(n);                  % 物理状态跟踪误差权重
R = 0.0001;                      % Delta-u 权重
deltaT = 0.05;
Nsim = 100;                     % 离线采样步数
Ntraj = 100;                    % 离线采样轨迹数
Nsteps = 10000;
SwitchStep = 100;                % 第 101 步起切换真实系统

x0 = [-2;-2];
reference = [1;0];
U0 = 0;
umin = -6;
umax = 6;
duMax = 3;
delta = 0.00001;                 % invK_G = pinv(delta*I)，不是初始 A/B
TerminalMode = 'LmiOnline';
LmiStateWeight = 800;            % 全增广状态 LMI 设计权重
ClassicalRK4 = false;            % false 保留现有实验的 k4=k1；true 用 k4=k3

MakePlots = true;
SaveResults = true;
PrintEvery = 1;
SdpVerbose = 0;
SdpTolerance = 1e-9;
PositiveMargin = 1e-8;
ResidualTolerance = 1e-6;

%% *************************** Dynamics ***********************************
f_u = @(t,x,u) [2*x(2,:);2*x(2,:)-10*x(1,:).^2.*x(2,:)-0.8*x(1,:)+u];
f_u_changed = @(t,x,u) [x(2,:);-3*x(2,:)-10*x(1,:).^2.*x(2,:)-3*x(1,:)+u];

%% ************************** Discretization ******************************
k1 = @(t,x,u) f_u(t,x,u);
k2 = @(t,x,u) f_u(t,x+k1(t,x,u)*deltaT/2,u);
k3 = @(t,x,u) f_u(t,x+k2(t,x,u)*deltaT/2,u);
if ClassicalRK4
    k4 = @(t,x,u) f_u(t,x+k3(t,x,u)*deltaT,u);
else
    k4 = @(t,x,u) f_u(t,x+k1(t,x,u)*deltaT,u);
end
f_ud = @(t,x,u) x+deltaT*(k1(t,x,u)+2*k2(t,x,u)+2*k3(t,x,u)+k4(t,x,u))/6;

k1_changed = @(t,x,u) f_u_changed(t,x,u);
k2_changed = @(t,x,u) f_u_changed(t,x+k1_changed(t,x,u)*deltaT/2,u);
k3_changed = @(t,x,u) f_u_changed(t,x+k2_changed(t,x,u)*deltaT/2,u);
if ClassicalRK4
    k4_changed = @(t,x,u) f_u_changed(t,x+k3_changed(t,x,u)*deltaT,u);
else
    k4_changed = @(t,x,u) f_u_changed(t,x+k1_changed(t,x,u)*deltaT,u);
end
f_ud_changed = @(t,x,u) x+deltaT*(k1_changed(t,x,u)+2*k2_changed(t,x,u)+2*k3_changed(t,x,u)+k4_changed(t,x,u))/6;

%% ************************** Paths and solver ****************************
root = fileparts(mfilename('fullpath'));
WeightFile = findWeights(root,'Good_VDP.mat',{'Correct','VDP','Ver1'});
assert(isfile(WeightFile),'Encoder weights not found: %s',WeightFile);
assert(exist('sdpvar','file')~=0 && exist('optimize','file')~=0,'YALMIP is required.');
assert(exist('mosekopt','file')~=0,'MOSEK is required.');
assert(any(strcmp(TerminalMode,{'Q','LqrFixed','LmiFixed','LmiOnline'})),'Unknown TerminalMode.');
validateattributes(N,{'numeric'},{'scalar','integer','positive'});
validateattributes(Nsteps,{'numeric'},{'scalar','integer','positive'});
validateattributes(PrintEvery,{'numeric'},{'scalar','integer','positive'});
validateattributes(Nsim,{'numeric'},{'scalar','integer','positive'});
validateattributes(Ntraj,{'numeric'},{'scalar','integer','positive'});
validateattributes(x0,{'double'},{'size',[2 1],'finite'});
validateattributes(reference,{'double'},{'size',[2 1],'finite'});
validateattributes(Q,{'double'},{'size',[2 2],'finite'});
assert(norm(Q-Q','fro')<1e-12 && min(eig(Q))>0,'Q must be symmetric positive definite.');
assert(all(isfinite([R delta deltaT duMax LmiStateWeight SdpTolerance PositiveMargin ResidualTolerance])) ...
    && all([R delta deltaT duMax LmiStateWeight SdpTolerance PositiveMargin ResidualTolerance]>0), ...
    'Weights, time step, delta and tolerances must be positive and finite.');
assert(isfinite(umin) && isfinite(umax) && umin<umax && U0>=umin && U0<=umax,'Invalid input settings.');
validateattributes(SwitchStep,{'numeric'},{'scalar','integer','nonnegative'});
% 只有数值求解的内部设置集中传给底部函数，实验参数直接在脚本开头修改。
LmiSettings = struct('TerminalMode',TerminalMode,'LmiStateWeight',LmiStateWeight, ...
    'StopOnLmiFailure',true,'SdpVerbose',SdpVerbose,'SdpTolerance',SdpTolerance, ...
    'PositiveMargin',PositiveMargin,'ResidualTolerance',ResidualTolerance);
settings = struct('TerminalMode',TerminalMode,'InitialState',x0,'Reference',reference, ...
    'InitialInput',U0,'Seed',Seed,'Steps',Nsteps,'Horizon',N,'StateWeight',Q, ...
    'IncrementWeight',R,'OnlineResetDelta',delta,'InputMin',umin,'InputMax',umax, ...
    'IncrementMax',duMax,'SampleTime',deltaT,'SwitchStep',SwitchStep, ...
    'TrainingSteps',Nsim,'TrainingTrajectories',Ntraj,'ClassicalRK4',ClassicalRK4, ...
    'WeightFile',WeightFile,'LmiSettings',LmiSettings);

%% ************************** Collect data ********************************
tic; disp('Starting data collection');
Ubig = 4*rand(Nsim,Ntraj)-2;
Xcurrent = 4*rand(n,Ntraj)-2;
X = zeros(n,Nsim*Ntraj); Y = X; U = zeros(1,Nsim*Ntraj);
for i = 1:Nsim
    Xnext = f_ud(0,Xcurrent,Ubig(i,:));
    columns = (i-1)*Ntraj+(1:Ntraj);
    X(:,columns) = Xcurrent; Y(:,columns) = Xnext;
    U(:,columns) = Ubig(i,:); Xcurrent = Xnext;
end
fprintf('Data collection DONE, time = %.2f s\n',toc);
%% ************************** Basis functions / Lift **********************
weights = load(WeightFile);
liftFun = @(x) localEncoder(x,weights);
Xlift = liftFun(X);
Ylift = liftFun(Y);
assert(size(Xlift,1)==Nlift,'This script requires the supplied 8-D encoder.');

%% ************************** Build predictor *****************************
disp('Starting REGRESSION'); tic;
V = [Xlift;U]; W = [Ylift;X];
M = (W*V')*pinv(V*V');
Abase = M(1:Nlift,1:Nlift); Bbase = M(1:Nlift,Nlift+1:end);
Cbase = M(Nlift+1:end,1:Nlift);
A = [Abase Bbase;zeros(m,Nlift) eye(m)];
B = [Bbase;eye(m)];
C = [Cbase zeros(n,m)];
fprintf('Regression done, time = %.2f s\n',toc);
% 增广状态 Lift_xu=[z;u_previous]，优化变量为 Delta-u。
na = Nlift+m;
Yr = repmat(reference,N,1);
R_bar = R*eye(N);
Lift_xu = [liftFun(x0);U0];
zReference = liftFun(reference);

results = struct('settings',settings,'status','running','completedSteps',0, ...
    'Q',Q,'R',R,'N',N,'deltaT',deltaT,'reference',reference, ...
    'training',struct('X',X,'Y',Y,'U',U),'weights',weights, ...
    'initialA',A,'initialB',B,'initialC',C, ...
    'sourceCode',fileread([mfilename('fullpath') '.m']), ...
    'LmiCertificateAppliesToMappedTerminalCost',false);
results.X = nan(n,Nsteps+1); results.X(:,1) = x0;
results.U = nan(1,Nsteps); results.DeltaU = results.U;
results.P = nan(na,na,Nsteps);
results.OutputTerminalWeight = nan(n,n,Nsteps);
results.A = nan(na,na,Nsteps); results.B = nan(na,m,Nsteps);
results.C = nan(n,na,Nsteps);
results.LmiSucceeded = false(1,Nsteps);
results.LmiInfo = cell(1,Nsteps); results.QpStatus = cell(1,Nsteps);
results.MpcCost = nan(1,Nsteps);
results.QpReturnCode = nan(1,Nsteps);
results.QpStationarityResidual = nan(1,Nsteps);
results.QpComplementarityResidual = nan(1,Nsteps);
folder = '';
if SaveResults
    parent = fullfile(root,'vdp_terminal_script_runs');
    if ~isfolder(parent), mkdir(parent); end
    folder = tempname(parent); mkdir(folder);
    results.runDirectory = folder;
end

try
    %% ************************** Initial MPC *****************************
    [P,Qterminal,info] = terminalWeight(A,B,C,Q,R,Lift_xu, ...
        zReference,umin,umax,LmiSettings);
    fixedP=P; fixedQterminal=Qterminal; fixedInfo=info;
    Q_bar = kron(eye(N),Q);
    Q_bar(end-n+1:end,end-n+1:end) = Qterminal; % 用 CPC' 替换最后一块 Q
    Compact_Form1 = zeros(n*N,na);
    Compact_Form2 = zeros(n*N,m*N);
    for ip = 1:N
        rows = (ip-1)*n+(1:n);
        Compact_Form1(rows,:) = C*A^ip;
        for jp = 1:ip
            columns = (jp-1)*m+(1:m);
            Compact_Form2(rows,columns) = C*A^(ip-jp)*B;
        end
    end
    [U0_Set,qp] = solveTrackingQp(Compact_Form1,Compact_Form2,Lift_xu,Yr, ...
        Q_bar,R_bar,U0,umin,umax,0.1);
    results.preliminary = struct('P',P,'outputTerminalWeight',Qterminal, ...
        'lmi',info,'qp',qp,'deltaU',U0_Set(1));
    % Preliminary input is not applied to the plant; keep InitialInput.
    Lift_xu = [liftFun(x0);U0];

    %% ************************** Online MPC ******************************
    K_A = zeros(Nlift,Nlift+m);
    invK_G = pinv(delta*eye(Nlift+m)); % Original delta*I initialization.
    bar_X = zeros(n,Nlift); bar_Q = 100*eye(Nlift);
    for i = 1:Nsteps
        if strcmp(TerminalMode,'LmiOnline')
            % 先用当前在线模型 optimize(Constraints,gamma)，再解 MPC。
            [P,Qterminal,info] = terminalWeight(A,B,C,Q,R,Lift_xu, ...
                zReference,umin,umax,LmiSettings);
        else
            P=fixedP; Qterminal=fixedQterminal; info=fixedInfo;
        end
        Q_bar = kron(eye(N),Q);
        Q_bar(end-n+1:end,end-n+1:end) = Qterminal; % 用 CPC' 替换最后一块 Q
        Compact_Form1 = zeros(n*N,na);
        Compact_Form2 = zeros(n*N,m*N);
        for ip = 1:N
            rows = (ip-1)*n+(1:n);
            Compact_Form1(rows,:) = C*A^ip;
            for jp = 1:ip
                columns = (jp-1)*m+(1:m);
                Compact_Form2(rows,columns) = C*A^(ip-jp)*B;
            end
        end
        [U0_Set,qp] = solveTrackingQp(Compact_Form1,Compact_Form2,Lift_xu,Yr, ...
            Q_bar,R_bar,U0,umin,umax,duMax);
        U0 = U0+U0_Set(1);
        x = x0;
        if i>SwitchStep
            x0 = f_ud_changed(0,x,U0);
        else
            x0 = f_ud(0,x,U0);
        end
        assert(all(isfinite(x0)),'VDPTrial:Plant','Nonfinite plant state.');

        results.X(:,i+1) = x0; results.U(i) = U0;
        results.DeltaU(i) = U0_Set(1); results.P(:,:,i) = P;
        results.OutputTerminalWeight(:,:,i) = Qterminal;
        results.A(:,:,i) = A; results.B(:,:,i) = B; results.C(:,:,i) = C;
        results.LmiSucceeded(i) = info.success; results.LmiInfo{i} = info;
        results.QpStatus{i} = qp.status; results.MpcCost(i) = qp.cost;
        results.QpReturnCode(i) = qp.returnCode;
        results.QpStationarityResidual(i) = qp.stationarityResidual;
        results.QpComplementarityResidual(i) = qp.complementarityResidual;
        results.completedSteps = i;

        % Same recursive A/B update as the supplied script, starting online.
        zx = liftFun(x); zy = liftFun(x0); v = [zx;U0];
        g = invK_G*v;
        invK_G = invK_G-(g*g')/(1+v'*g);
        K_A = K_A+zy*v';
        Kext = K_A*invK_G;
        Abase = Kext(:,1:Nlift); Bbase = Kext(:,Nlift+1:end);
        % Same recursive output-map update; C is NOT held fixed.
        bar_X = bar_X+x*zx';
        gC = bar_Q*zx;
        bar_Q = bar_Q-(gC*gC')/(1+zx'*gC);
        Cbase = bar_X*bar_Q;
        A = [Abase Bbase;zeros(m,Nlift) eye(m)];
        B = [Bbase;eye(m)];
        C = [Cbase zeros(n,m)];
        Lift_xu = [liftFun(x0);U0];
        if i==1 || mod(i,PrintEvery)==0 || i==Nsteps
            fprintf('k=%5d, x1=% .8f, u=% .6f, error=% .3e, LMI=%d\n', ...
                i,x0(1),U0,x0(1)-reference(1),info.success);
        end
    end
catch exception
    results.status = 'failed';
    results.failure = struct('identifier',exception.identifier, ...
        'message',exception.message,'A',A,'B',B,'C',C,'Lift_xu',Lift_xu);
    if SaveResults
        save(fullfile(folder,'failed_run.mat'),'results','-v7.3');
        fprintf('Failed-run diagnostics saved: %s\n',folder);
    end
    rethrow(exception)
end
results.status = 'completed';
results.Steady_Error = abs(results.X(1,end-1)-reference(1));
results.FinalStateError = abs(results.X(1,end)-reference(1));
fprintf('Steady_Error (original index) = %.12g\n',results.Steady_Error);
fprintf('FinalStateError = %.12g\n',results.FinalStateError);
tail=min(100,Nsteps); errTail=results.X(1,end-tail+1:end)-reference(1);
results.metrics=struct('FinalPositionError',results.FinalStateError, ...
    'TailPositionRMSE',sqrt(mean(errTail.^2)), ...
    'MaximumAbsoluteInput',max(abs(results.U)), ...
    'LmiSteps',nnz(results.LmiSucceeded),'CompletedSteps',results.completedSteps);
disp(results.metrics);
if SaveResults
    save(fullfile(folder,'results.mat'),'results','-v7.3');
    fprintf('Saved: %s\n',fullfile(folder,'results.mat'));
end
%% ************************** Results / Plots *****************************
X_Collection = results.X(:,1:end-1); % 每次施加控制前的真实状态
U_Collection = results.U;
Steady_Error = results.Steady_Error; 
FinalStateError = results.FinalStateError; % 第 Nsteps 次推进后的误差
if MakePlots
    time = (0:Nsteps)*deltaT;
    figure
    plot(time,results.X(1,:),'LineWidth',2)
    yline(reference(1),'--')
    xlabel('Time (s)'); ylabel('x_1'); grid on
    figure
    plot(time,results.X(2,:),'LineWidth',2)
    yline(reference(2),'--')
    xlabel('Time (s)'); ylabel('x_2'); grid on
    figure
    stairs(time(1:end-1),U_Collection,'LineWidth',2)
    yline(umin,'--'); yline(umax,'--')
    xlabel('Time (s)'); ylabel('u'); grid on
end

%% ************************** Local functions *****************************
function z = localEncoder(x,w)
% Same encoder, loaded once from an explicit path; relu function.
h1 = max(w.W1*x+w.b1(:),0);
h2 = max(w.W2*h1+w.b2(:),0);
h3 = max(w.W3*h2+w.b3(:),0);
z = w.W4*h3+w.b4(:);
end

function [P,Qterminal,info] = terminalWeight(A,B,C,Q,R,eta,zr,umin,umax,cfg)
try
    Bbase=B(1:end-1,:); Abase=A(1:end-1,1:end-1);
    if norm(Bbase)<=eps
        ur=0;
    else
        ur=(Bbase'*(zr-Abase*zr))/(Bbase'*Bbase);
    end
    ur=min(max(ur,umin),umax);
    e=eta-[zr;ur];
    % Full augmented-state design weight; shared by LQR/LMI comparisons.
    Qlmi=cfg.LmiStateWeight*eye(size(A));
    if strcmp(cfg.TerminalMode,'Q')
        P=nan(size(A)); Qterminal=Q;
        info=struct('success',false,'gamma',nan,'minimumFullPDecay',nan, ...
            'F',nan(1,size(A,1)),'uReference',ur,'problem',nan,'message','Ordinary Q baseline');
        return
    elseif strcmp(cfg.TerminalMode,'LqrFixed')
        [K,P]=dlqr(A,B,Qlmi,R); F=-K; gamma=e'*P*e; minD=nan;
        numerics=struct('scalingSeedMethod','not used');
    else
        [P,F,gamma,minD,numerics]=solveScaledTerminalLmi(A,B,Qlmi,R,e,cfg);
    end
    Qterminal=C*P*C'; Qterminal=(Qterminal+Qterminal')/2;
    assert(all(isfinite(Qterminal(:))),'VDPTrial:MappedP','Nonfinite CPC-transpose.');
    info=struct('success',~strcmp(cfg.TerminalMode,'LqrFixed'),'gamma',gamma,'minimumFullPDecay',minD, ...
        'F',F,'uReference',ur,'problem',0,'message','','numerics',numerics);
catch exception
    if cfg.StopOnLmiFailure, rethrow(exception); end
    % Explicit opt-in fallback. This is ordinary Q, NOT an LMI solution.
    P=nan(size(A)); Qterminal=Q;
    token=regexp(exception.message,'problem=(-?\d+)','tokens','once');
    problem=nan; if ~isempty(token), problem=str2double(token{1}); end
    info=struct('success',false,'gamma',nan,'minimumFullPDecay',nan, ...
        'F',nan(1,size(A,1)),'uReference',nan,'problem',problem,'message',exception.message);
    warning('VDPTrial:Fallback','Using ordinary terminal Q: %s',exception.message);
end
end

function [du,info] = solveTrackingQp(F,G,eta,Yr,Qbar,Rbar,uPrevious,umin,umax,duMax)
% Same first-input-only absolute bounds as supplied; N increment bounds.
% MOSEK takes 0.5*du'*Qqp*du+c'*du, so Qqp=2*H.
N=size(G,2); free=F*eta-Yr;
H=G'*Qbar*G+Rbar; H=(H+H')/2;
f=2*G'*Qbar*free;
scale=max([1,norm(H,inf),norm(f,inf)]);
[row,col,val]=find(tril(sparse(2*H/scale)));
Acons=[1 zeros(1,N-1);-1 zeros(1,N-1)];
bcons=[umax-uPrevious;uPrevious-umin];
prob=struct('qosubi',row,'qosubj',col,'qoval',val,'c',f/scale, ...
    'a',sparse(Acons),'blc',-inf(2,1),'buc',bcons, ...
    'blx',-duMax*ones(N,1),'bux',duMax*ones(N,1));
params=struct('MSK_DPAR_INTPNT_QO_TOL_PFEAS',1e-10, ...
    'MSK_DPAR_INTPNT_QO_TOL_DFEAS',1e-10,'MSK_DPAR_INTPNT_QO_TOL_REL_GAP',1e-10);
[code,answer]=mosekopt('minimize echo(0)',prob,params);
status='NO_SOLUTION';
if isfield(answer,'sol') && isfield(answer.sol,'itr'), status=answer.sol.itr.solsta; end
% checks below. Do not accept unknown/infeasible solution statuses.
if ~ismember(code,[0 100006]) || ~strcmpi(status,'OPTIMAL')
    error('VDPTrial:QpFailure','MOSEK QP code=%d, status=%s.',code,status);
end
du=answer.sol.itr.xx;
if any(~isfinite(du)) || numel(du)~=N || ...
        max([Acons*du-bcons;abs(du)-duMax])>1e-6
    error('VDPTrial:QpResidual', ...
        'QP input-bound residual failed: max violation %.12g, du(1)=%.12g.', ...
        max([Acons*du-bcons;abs(du)-duMax]),du(1));
end
s=answer.sol.itr;
gradient=(2*H*du+f)/scale;
dualTerm=Acons'*(s.suc-s.slc)-s.slx+s.sux;
stationarity=norm(gradient+dualTerm,inf)/max([1,norm(gradient,inf),norm(dualTerm,inf)]);
complementarity=max(abs([s.slx.*(du+duMax);s.sux.*(duMax-du); ...
    s.suc.*(bcons-Acons*du)]))/max(1,abs((du'*H*du+f'*du)/scale));
dualMinimum=min([s.slx;s.sux;s.slc;s.suc]);
if stationarity>1e-6 || complementarity>1e-6 || dualMinimum < -1e-8
    error('VDPTrial:QpKkt', ...
        'QP KKT checks failed: stationarity %.6g, complementarity %.6g, dual %.6g.', ...
        stationarity,complementarity,dualMinimum);
end
info=struct('status',status,'cost',du'*H*du+f'*du+free'*Qbar*free, ...
    'returnCode',code,'stationarityResidual',stationarity, ...
    'complementarityResidual',complementarity);
end

function [P,F,gammaValue,minD,details] = solveScaledTerminalLmi(A,B,Q,R,e,cfg)
n=size(A,1); Q=(Q+Q')/2;
Qseed=Q+1e-6*eye(n); Pseed=[]; seedMethod='finite iteration';
if exist('idare','file')~=0
    [candidate,~,~,seedInfo]=idare(A,B,Qseed,R,[],[]);
    if seedInfo.Report==0 && ~isempty(candidate) && all(isfinite(candidate(:)))
        candidate=(candidate+candidate')/2;
        [~,flag]=chol(candidate);
        if flag==0, Pseed=candidate; seedMethod='idare'; end
    end
end
if isempty(Pseed)
    Pseed=Qseed;
    for j=1:500
        Kseed=-((R+B'*Pseed*B)\(B'*Pseed*A));
        Ac=A+B*Kseed;
        Pseed=Qseed+R*(Kseed'*Kseed)+Ac'*Pseed*Ac;
        Pseed=(Pseed+Pseed')/2;
    end
end
Kseed=-((R+B'*Pseed*B)\(B'*Pseed*A));
costScale=max(1,e'*Pseed*e);
T=sqrt(costScale)*(chol(Pseed)\eye(n));
inputScale=sqrt(costScale/R);
As=T\(A*T); Bs=(T\B)*inputScale; es=T\e;
Ks=Kseed*T/inputScale;
% Compact factor for semidefinite Q; retain all strictly positive directions.
[V,D]=eig(Q); d=diag(D); keep=d>max(1,max(abs(d)))*1e-12;
Lq=diag(sqrt(d(keep)))*V(:,keep)'*T/sqrt(costScale);
nr=size(Lq,1); lr=sqrt(R/costScale)*inputScale;
S=sdpvar(n,n,'symmetric'); Yd=sdpvar(1,n,'full'); gamma=sdpvar(1,1);
Y=Yd+Ks*S;
transition=(As+Bs*Ks)*S+Bs*Yd;
decay=[S transition' (Lq*S)' (lr*Y)'; ...
    transition S zeros(n,nr) zeros(n,1); ...
    Lq*S zeros(nr,n) gamma*eye(nr) zeros(nr,1); ...
    lr*Y zeros(1,n) zeros(1,nr) gamma];
invT=T\eye(n);
constraints=[S>=cfg.PositiveMargin*(invT*invT'), ...
    gamma>=cfg.PositiveMargin/costScale,[1 es';es S]>=0,decay>=0];
ops=sdpsettings('solver','mosek','verbose',cfg.SdpVerbose, ...
    'mosek.MSK_DPAR_INTPNT_CO_TOL_PFEAS',cfg.SdpTolerance, ...
    'mosek.MSK_DPAR_INTPNT_CO_TOL_DFEAS',cfg.SdpTolerance, ...
    'mosek.MSK_DPAR_INTPNT_CO_TOL_REL_GAP',cfg.SdpTolerance);
sol=optimize(constraints,gamma,ops);
if sol.problem~=0
    error('VDPTrial:SdpFailure','Scaled LMI problem=%d: %s',sol.problem,sol.info);
end
Sv=value(S); Sv=(Sv+Sv')/2; Yv=value(Yd); gv=value(gamma);
if any(~isfinite(Sv(:))) || any(~isfinite(Yv(:))) || ~isfinite(gv) || gv<=0
    error('VDPTrial:InvalidLmi','Nonfinite/invalid LMI decision values.');
end
[~,flag]=chol(Sv);
if flag~=0 || rcond(Sv)<1e-12
    error('VDPTrial:LmiConditioning','Scaled Q1 is not reliably positive definite.');
end
P=costScale*(T'\((gv*(Sv\eye(n)))/T)); P=(P+P')/2;
F=inputScale*((Yv/Sv+Ks)/T); gammaValue=costScale*gv;
Acl=A+B*F; D=P-Acl'*P*Acl-Q-R*(F'*F); D=(D+D')/2;
minD=min(eig(D));
scale=max([1,norm(P,2),norm(Acl'*P*Acl,2),norm(Q,2),norm(R*(F'*F),2)]);
stateMargin=min(eig([1 es';es Sv]));
tol=cfg.ResidualTolerance;
if min(eig(P))<=0 || minD < -tol*scale || ...
        stateMargin < -tol*max(1,norm(Sv,2))
    error('VDPTrial:LmiResidual','Original-coordinate decay check failed: minD=%.6g.',minD);
end
details=struct('relativeDecayResidual',minD/scale,'conditionS',rcond(Sv), ...
    'costScale',costScale,'coordinateCondition',cond(T),'solverInfo',sol.info, ...
    'scalingSeedMethod',seedMethod);
end

function path = findWeights(folder,name,parts)
path = fullfile(folder,name);
if isfile(path), return; end
while true
    candidate = fullfile(folder,parts{:},name);
    if isfile(candidate), path=candidate; return; end
    parent = fileparts(folder);
    if isempty(parent) || strcmp(parent,folder), break; end
    folder = parent;
end
end
