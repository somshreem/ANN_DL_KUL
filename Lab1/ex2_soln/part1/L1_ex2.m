x_original = 0 : 0.05 : 3*pi;
y_original = sin(x_original.^2);
x=con2seq(x_original); y=con2seq(y_original); % convert the data to a useful format
num_of_neurons = 20; 

net_gd = feedforwardnet(num_of_neurons,'traingd');
net_lm = feedforwardnet(num_of_neurons,'trainlm');
net_bfg = feedforwardnet(num_of_neurons,'trainbfg');

net_gd.iw{1,1}=net_lm.iw{1,1};  %set the same weights for the networks //lm
net_gd.lw{2,1}=net_lm.lw{2,1};

net_gd.iw{1,1}=net_bfg.iw{1,1};  %set the same weights for the networks //bfg
net_gd.lw{2,1}=net_bfg.lw{2,1};

net_gd.b{1}=net_lm.b{1}; %set the same biases for the networks //lm
net_gd.b{2}=net_lm.b{2};

net_gd.b{1}=net_bfg.b{1}; %set the same biases for the networks //bfg
net_gd.b{2}=net_bfg.b{2};

%training

net_gd.trainParam.epochs=1;  % set the number of epochs for the training =1
net_lm.trainParam.epochs=1;
net_bfg.trainParam.epochs=1;

net_gd = train(net_gd,x,y);
net_lm = train(net_lm,x,y);
net_bfg = train(net_bfg,x,y);

net_out_gd_1 = net_gd(x);
net_out_lm_1 = net_lm(x);
net_out_bfg_1 = net_bfg(x);


net_gd.trainParam.epochs=50;  % set the number of epochs for the training =50
net_lm.trainParam.epochs=50;
net_bfg.trainParam.epochs=50;

net_gd = train(net_gd,x,y);
net_lm = train(net_lm,x,y);
net_bfg = train(net_bfg,x,y);

net_out_gd_2 = net_gd(x);
net_out_lm_2 = net_lm(x);
net_out_bfg_2 = net_bfg(x);


net_gd.trainParam.epochs=1000;  % set the number of epochs for the training =1000
net_lm.trainParam.epochs=1000;
net_bfg.trainParam.epochs=1000;

net_gd = train(net_gd,x,y);
net_lm = train(net_lm,x,y);
net_bfg = train(net_bfg,x,y);

net_out_gd_3 = net_gd(x);
net_out_lm_3 = net_lm(x);
net_out_bfg_3 = net_bfg(x);


%plots : gd vs lm 
figure
subplot(3,1,1);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_1),'r',x_original,cell2mat(net_out_lm_1),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','traingd','trainlm','Location','north');


% subplot(3,3,2);
% postregm(cell2mat(net_out_gd_1),y_original); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm(cell2mat(net_out_lm_1),y_original);

subplot(3,1,2);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_2),'r',x_original,cell2mat(net_out_lm_2),'g');
title('50 epochs');
legend('target','traingd','trainlm','Location','north');


% subplot(3,3,5);
% postregm(cell2mat(net_out_gd_2),y_original);
% subplot(3,3,6);
% postregm(cell2mat(net_out_lm_2),y_original);

subplot(3,1,3);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_3),'r',x_original,cell2mat(net_out_lm_3),'g');
title('1000 epochs');
legend('target','traingd','trainlm','Location','north');
% subplot(3,3,8);
% postregm(cell2mat(net_out_gd_3),y_original);
% subplot(3,3,9);
% postregm(cell2mat(net_out_lm_3),y_original);
saveas(gcf, 'img/gd_vs_lm_1_50_1000.png')



%plots : gd vs bfg
figure
subplot(3,1,1);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_1),'r',x_original,cell2mat(net_out_bfg_1),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','traingd','trainbfg','Location','north');
% subplot(3,3,2);
% postregm(cell2mat(net_out_gd_1),y_original); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm(cell2mat(net_out_bfg_1),y_original);

subplot(3,1,2);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_2),'r',x_original,cell2mat(net_out_bfg_2),'g');
title('50 epochs');
legend('target','traingd','trainbfg','Location','north');
% subplot(3,3,5);
% postregm(cell2mat(net_out_gd_2),y_original);
% subplot(3,3,6);
% postregm(cell2mat(net_out_bfg_2),y_original);

subplot(3,1,3);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_3),'r',x_original,cell2mat(net_out_bfg_3),'g');
title('1000 epochs');
legend('target','traingd','trainbfg','Location','north');
% subplot(3,3,8);
% postregm(cell2mat(net_out_gd_3),y_original);
% subplot(3,3,9);
% postregm(cell2mat(net_out_bfg_3),y_original);
saveas(gcf, 'img/gd_vs_bfg_1_50_1000.png')




%plots : gd vs bfg vs lm
figure
subplot(3,1,1);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_1),'r',x_original,cell2mat(net_out_bfg_1),'g', x_original,cell2mat(net_out_lm_1),'m'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','traingd','trainbfg','trainlm','Location','north');
% subplot(3,3,2);
% postregm(cell2mat(net_out_gd_1),y_original); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm(cell2mat(net_out_bfg_1),y_original);

subplot(3,1,2);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_2),'r',x_original,cell2mat(net_out_bfg_2),'g', x_original,cell2mat(net_out_lm_2),'m');
title('50 epochs');
legend('target','traingd','trainbfg','trainlm','Location','north');
% subplot(3,3,5);
% postregm(cell2mat(net_out_gd_2),y_original);
% subplot(3,3,6);
% postregm(cell2mat(net_out_bfg_2),y_original);

subplot(3,1,3);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_gd_3),'r',x_original,cell2mat(net_out_bfg_3),'g', x_original,cell2mat(net_out_lm_3),'m');
title('1000 epochs');
legend('target','traingd','trainbfg','trainlm','Location','north');
% subplot(3,3,8);
% postregm(cell2mat(net_out_gd_3),y_original);
% subplot(3,3,9);
% postregm(cell2mat(net_out_bfg_3),y_original);
saveas(gcf, 'img/gd_vs_bfg_vs_lm_1_50_1000.png')
