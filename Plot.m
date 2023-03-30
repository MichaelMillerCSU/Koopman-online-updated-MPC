clc
clear
% close all

% load DuffingPlot_trajectory
% plotTime = 100;
% tspan_pred = tspan_pred(1 : plotTime);
% marker_T = marker_T(1 : plotTime / 10);
% marker_originX = marker_originX(:, 1 : plotTime / 10);
% Uplot = Uplot(:, 1 : plotTime);
% % for i = 1: 2
% %     subplot(2, 1, i)
% %     plot(tspan_pred, X(i, 1 :plotTime));
% %     hold on
% %     plot(tspan_pred, test_Y(i, 1 :plotTime));
% %     hold on
% %     scatter(marker_T, marker_originX(i, :), 30,'filled');
% %     hold off
% % end
% 
% % plot(tspan_pred, Uplot)
% N = 100;
% N_traj = 100;
% scatter(X(1, :), X(2, :), 10, 'filled')



% load DuffingPlot
% plotTime = 10000;
% 
% logXloc_collection = logXloc(:, 1 : plotTime);
% logXLOClift_collection = logXLOClift(:, 1 : plotTime);
% logU_collection = logUloc(:, 1 : plotTime);
% A_error_collection = A_error(:, 1 : plotTime);
% B_error_collection = B_error(:, 1 : plotTime);
% C_error_collection = C_error(:, 1 : plotTime);

% load DuffingPlotrealtime
% tspan = tspan(1 : plotTime);
% plot(tspan, logX(1, 1 : plotTime))
% hold on 
% plot(tspan, logXloc_collection(1, :))
% hold on 
% plot(tspan, logXloc(1, 1 : plotTime))
% hold on 
% plot(tspan, logR(1, 1 : plotTime))

% figure
% plot(tspan, logU(1, 1 : plotTime))
% hold on 
% plot(tspan, logU_collection(1, :))
% hold on 
% plot(tspan, logUloc(1, 1 : plotTime))

% figure
% plot(tspan, A_error_collection(1, 1 : plotTime))
% hold on
% plot(tspan, B_error_collection(1, 1 : plotTime))
% hold on
% plot(tspan, C_error_collection(1, 1 : plotTime))
% 
% 
% figure
% plot(tspan, A_error(1, 1 : plotTime))
% hold on
% plot(tspan, B_error(1, 1 : plotTime))
% hold on
% plot(tspan, C_error(1, 1 : plotTime))

% load vdpPlot_trajectory
% figure
% plotTime = 100;
% tspan_pred = tspan_pred(1 : plotTime);
% marker_T = marker_T(1 : plotTime / 10);
% marker_originX = marker_originX(:, 1 : plotTime / 10);
% Uplot = Uplot(:, 1 : plotTime);
% for i = 1: 2
%     subplot(2, 1, i)
%     plot(tspan_pred, X(i, 1 :plotTime));
%     hold on
%     plot(tspan_pred, test_Y(i, 1 :plotTime));
%     hold on
%     scatter(marker_T, marker_originX(i, :), 30,'filled');
%     hold off
% end
% plot(tspan_pred, Uplot);

% figure
% scatter(X(1, :), X(2, :), 10, 'filled')


load VDPPlot
plotTime = 10000;

logXloc_collection = logXloc(:, 1 : plotTime);
logXLOClift_collection = logXLOClift(:, 1 : plotTime);
logU_collection = logUloc(:, 1 : plotTime);
A_error_collection = A_error(:, 1 : plotTime);
B_error_collection = B_error(:, 1 : plotTime);
C_error_collection = C_error(:, 1 : plotTime);

load VDPPlotrealtime
tspan = tspan(1 : plotTime);
figure
plot(tspan, logX(1, 1 : plotTime))
hold on 
plot(tspan, logXloc_collection(1, :))
hold on 
plot(tspan, logXloc(1, 1 : plotTime))
hold on 
plot(tspan, logR(1, 1 : plotTime))

figure
plot(tspan, logU(1, 1 : plotTime))
hold on 
plot(tspan, logU_collection(1, :))
hold on 
plot(tspan, logUloc(1, 1 : plotTime))
% 
% figure
% plot(tspan, A_error_collection(1, 1 : plotTime))
% hold on
% plot(tspan, B_error_collection(1, 1 : plotTime))
% hold on
% plot(tspan, C_error_collection(1, 1 : plotTime))


% figure
% plot(tspan, A_error(1, 1 : plotTime))
% hold on
% plot(tspan, B_error(1, 1 : plotTime))
% hold on
% plot(tspan, C_error(1, 1 : plotTime))


% figure
% plotTime = 2000;
% for i = 1: 8
%     subplot(2, 4, i);
%     plot(tspan(1:plotTime), logXLOClift(i, 1:plotTime))
%     hold on
%     plot(tspan(1:plotTime), logXLOClift_collection(i, 1:plotTime))
%     hold on
%     plot(tspan(1:plotTime), logRlift(i, 1:plotTime));
%     hold on
% end

% figure
% plotTime = 2000;
% 
% for i = 1: 8
%     plot(tspan(1:plotTime), logXLOClift(i, 1:plotTime))
%     hold on
%     plot(tspan(1:plotTime), logXLOClift_collection(i, 1:plotTime))
%     hold on
%     plot(tspan(1:plotTime), logRlift(i, 1:plotTime));
%     hold on
% end




