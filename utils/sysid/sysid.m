% using the system identification toolbox

% load mat file
load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/roll_rate_test.mat');

% create data
y = xs_true;
u = us;
frequency = 500;

% append zeros before u and y
y = [zeros(1,20),y].';
u = [zeros(1,20),u].';
t = 0:1/frequency:(length(y)-1)/frequency;

% create idddata from y and u
data = iddata(y,u,1/frequency);

% create model with tfest
model = tfest(data,3,0);
% get poles and zeros
[z,p,k] = zpkdata(model);

% using model to predict y
% load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/roll_rate_test2.mat');
% u = us2.'; y = xs_true2.';
load('/home/pcy/Research/code/crazyswarm2-adaptive/utils/crazyflie-firmware-adaptive/tools/usdlog/roll_rate_test3.mat');
u = us3.'; y = xs_true3.';
y_predict = lsim(model, u, 0:1/frequency:(length(u)-1)/frequency);
% y = xs_true3;
% u = us3;

% plot y and y_predict
figure(1)
plot(y)
hold on
plot(y_predict)
plot(u)
legend('y','y_{predict}','u')
hold off
% save figure
saveas(gcf,'roll_rate_test.png')