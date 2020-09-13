x_original = 0 : 0.05 : 3*pi;
y_original = sin(x_original.^2);
x=con2seq(x_original); y=con2seq(y_original); % convert the data to a useful format
num_of_neurons = 20; 

net_gd = feedforwardnet(num_of_neurons,'traingd');
net_lm = feedforwardnet(num_of_neurons,'trainlm');
net_br = feedforwardnet(num_of_neurons,'trainbr');

net_br.iw{1,1}=net_lm.iw{1,1};  %set the same weights for the networks //lm
net_br.lw{2,1}=net_lm.lw{2,1};

net_br.iw{1,1}=net_gd.iw{1,1};  %set the same weights for the networks //gd
net_br.lw{2,1}=net_gd.lw{2,1};

net_br.b{1}=net_lm.b{1}; %set the same biases for the networks //lm
net_br.b{2}=net_lm.b{2};

net_br.b{1}=net_gd.b{1}; %set the same biases for the networks //gd
net_br.b{2}=net_gd.b{2};

%training

net_gd.trainParam.epochs=1;  % set the number of epochs for the training =1
net_lm.trainParam.epochs=1;
net_br.trainParam.epochs=1;

net_gd = train(net_gd,x,y);
net_lm = train(net_lm,x,y);
net_br = train(net_br,x,y);

net_out_gd_1 = net_gd(x);
net_out_lm_1 = net_lm(x);
net_out_br_1 = net_br(x);


net_gd.trainParam.epochs=50;  % set the number of epochs for the training =50
net_lm.trainParam.epochs=50;
net_br.trainParam.epochs=50;

net_gd = train(net_gd,x,y);
net_lm = train(net_lm,x,y);
net_br = train(net_br,x,y);

net_out_gd_2 = net_gd(x);
net_out_lm_2 = net_lm(x);
net_out_br_2 = net_br(x);


net_gd.trainParam.epochs=1000;  % set the number of epochs for the training =1000
net_lm.trainParam.epochs=1000;
net_br.trainParam.epochs=1000;

net_gd = train(net_gd,x,y);
net_lm = train(net_lm,x,y);
net_br = train(net_br,x,y);

net_out_gd_3 = net_gd(x);
net_out_lm_3 = net_lm(x);
net_out_br_3 = net_br(x);


%plots : br vs lm 
figure
%subplot(3,3,1);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_br_1),'r',x_original,cell2mat(net_out_lm_1),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainbr','trainlm','Location','north');
%subplot(3,3,2);
figure
postregm(cell2mat(net_out_br_1),y_original); % perform a linear regression analysis and plot the result
%subplot(3,3,3);
figure
postregm(cell2mat(net_out_lm_1),y_original);

%subplot(3,3,4);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_br_2),'r',x_original,cell2mat(net_out_lm_2),'g');
title('50 epochs');
legend('target','trainbr','trainlm','Location','north');
%subplot(3,3,5);
figure
postregm(cell2mat(net_out_br_2),y_original);
%subplot(3,3,6);
figure
postregm(cell2mat(net_out_lm_2),y_original);
figure
%subplot(3,3,7);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_br_3),'r',x_original,cell2mat(net_out_lm_3),'g');
title('1000 epochs');
legend('target','trainbr','trainlm','Location','north');
saveas(gcf, 'img/trainbr_vs_lm_1000.png')

figure
%subplot(3,3,8);
postregm(cell2mat(net_out_br_3),y_original);
figure
%subplot(3,3,9);
postregm(cell2mat(net_out_lm_3),y_original);




%plots : br vs gd
figure
%subplot(3,3,1);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_br_1),'r',x_original,cell2mat(net_out_gd_1),'g'); % plot the sine function and the output of the networks
title('1 epoch');
legend('target','trainbr','traingd','Location','north');
%subplot(3,3,2);
figure
postregm(cell2mat(net_out_br_1),y_original); % perform a linear regression analysis and plot the result
%subplot(3,3,3);
figure
postregm(cell2mat(net_out_gd_1),y_original);

figure
%subplot(3,3,4);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_br_2),'r',x_original,cell2mat(net_out_gd_2),'g');
title('50 epochs');
legend('target','trainbr','traingd','Location','north');
%subplot(3,3,5);
figure
postregm(cell2mat(net_out_br_2),y_original);
%subplot(3,3,6);
figure
postregm(cell2mat(net_out_gd_2),y_original);

figure
%subplot(3,3,7);
plot(x_original,y_original,'bx',x_original,cell2mat(net_out_br_3),'r',x_original,cell2mat(net_out_gd_3),'g');
title('1000 epochs');
legend('target','trainbr','traingd','Location','north');
saveas(gcf, 'img/trainbr_vs_gd_1000.png')

figure
%subplot(3,3,8);
postregm(cell2mat(net_out_br_3),y_original);
%subplot(3,3,9);
figure
postregm(cell2mat(net_out_gd_3),y_original);
