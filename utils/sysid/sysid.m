% using the system identification toolbox

% load mat file
load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/data_x.mat');

% create data
y = xs_true_x;
u = us_x;
frequency = 500;
substep = 10;

% append zeros before u and y
y = [zeros(1,20),y].';
u = [zeros(1,20),u].';
t = 0:1/frequency:(length(y)-1)/frequency;
y_sampled = y(1:substep:end);
u_sampled = u(1:substep:end);
t_sampled = t(1:substep:end);
frequency_sampled = frequency/substep;

% create idddata from y and u
% data = iddata(y,u,1/frequency);
data = iddata(y_sampled,u_sampled,1/frequency_sampled);

% create model with tfest
model = tfest(data,2,1);
% get poles and zeros
[z,p,k] = zpkdata(model);
p

% using model to predict y
% load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/roll_rate_test2.mat');
% u = us2.'; y = xs_true2.';
% load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/roll_rate_test3.mat');
% u = us3.'; y = xs_true3.';
load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/data_xy.mat');
u = us_xy; y = xs_true_xy; t = ts_xy;
u_sampled = u(1:substep:end);
y_sampled = y(1:substep:end);
t_sampled = t(1:substep:end); 

y_predict = lsim(model, u_sampled, 0:1/frequency_sampled:(length(u_sampled)-1)/frequency_sampled);
% y = xs_true3;
% u = us3;

% plot y and y_predict, make the figure wide
figure('Position',[100 100 1000 400])
plot(t_sampled, y_sampled, '-*')
hold on
% plot with dot
plot(t_sampled, y_predict, '-o')
plot(t_sampled, u_sampled)
legend('y','y_{predict}','u')
hold off
% save figure
saveas(gcf,'roll_rate_test.png')