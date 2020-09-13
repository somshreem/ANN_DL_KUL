%approximate a nonlinear func using feedeforward artificial neural network

% importing given data file
data_file = load('Data_Problem1_regression.mat');

X1 = data_file.X1;
X2 = data_file.X2;

T1 = data_file.T1;
T2 = data_file.T2;
T3 = data_file.T3;
T4 = data_file.T4;
T5 = data_file.T5;

% student number is r0813317
d1 = 8; d2 = 7; d3 = 3; d4 = 3; d5 = 1;

T_new = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5) / (d1+d2+d3+d4+d5); 

total_datapoints = 1:13600; %given

random_datapoints = randperm(length(total_datapoints));

T_training = T_new(random_datapoints(1:1000));

T_validation = T_new(random_datapoints(1001:2000));

T_test = T_new(random_datapoints(2001:3000));

X1_training = X1(random_datapoints(1:1000));

X2_training = X2(random_datapoints(1:1000));

X1_validation = X1(random_datapoints(1001:2000));

X2_validation = X2(random_datapoints(1001:2000));

X1_test = X1(random_datapoints(2001:3000));

X2_test = X2(random_datapoints(2001:3000));


%plot surface of training set

X1_linear_train = linspace(min(X1_training), max(X1_training), length(X1_training));
X2_linear_train = linspace(min(X2_training), max(X2_training), length(X2_training));
[X1_grid, X2_grid] = meshgrid(X1_linear_train, X2_linear_train);

figure
F = scatteredInterpolant(X1_training, X2_training, T_training)
V = F(X_grid, Y_grid);


surf(X_linear, Y_linear, V, 'EdgeColor','none');
saveas(gcf, 'img/surfaceplot_training_gd.png')

%figure
%scatter3(X1_rand, X2_rand, T_training)
%saveas(gcf, 'img/scatterplot_training.png')

%figure
%plot3(X_grid, Y_grid, V)
%saveas(gcf, 'img/threeD_training.png')

%figure
%mesh(X_grid, Y_grid, V)
%saveas(gcf, 'img/meshplot_training.png')


%build and train feedforward neural network



%X1_seq = con2seq(X1_random);
%X2_seq = con2seq(X2_random);
%Ttrain_seq = con2seq(Ttrain);

%p = [X1_seq; X2_seq];
%t = Ttrain_seq;

% net = train(net, p, t);
% view(net)
% 
% a = sim(net, p);

X = [X1_training,X2_training;
     X1_validation,X2_validation;
     X1_test,X2_test];
    
Y = [T_training;
    T_validation; 
    T_test];

X_input = X';
Y_input = Y';

results_matrix = zeros;
training_results =struct('traingd',[]);

hiddenSizes = 5;
train_algo = 'traingd';
epoch=500
num_of_neurons = 20;
transfer_functions = 'tansig';


net = feedforwardnet(num_of_neurons, train_algo);
net.trainParam.epochs = epoch;
net.layers{1}.transferFcn = transfer_functions;

net.divideFcn = 'divideblock';
net.divideParam.trainRatio = 1/3;
net.divideParam.valRatio = 1/3;
net.divideParam.testRatio = 1/3;

[net,tr] = train(net, X_input, Y_input);

valIndex = tr.valInd;
testIndex = tr.testInd;
traininigIndex = tr.trainInd;

training_results = setfield(training_results,train_algo,tr)

%predictions: 
            
predictions_train = (sim(net, X_input(:,traininigIndex)));
predictions_val = (sim(net,X_input(:,valIndex)));
predictions_test = (sim(net,X_input(:,testIndex)));
%plot(predictions_test, 'b');
%hold on
%plot(Y_input(:,testIndex), 'r');
%legend('train', 'test');
            
%Measures

%MSE = mean((pred-target).^2);

mse_train = mean(((predictions_train)-(Y_input(:,traininigIndex))).^2)
mse_val = mean(((predictions_val)-(Y_input(:,valIndex))).^2)
mse_test = mean(((predictions_test)-(Y_input(:,testIndex))).^2)
            
[m,b,r_train]=postregm((predictions_train),Y(traininigIndex,:)');
[m,b,r_test]=postregm((predictions_test),Y(testIndex,:)');
            
results_matrix=[mse_train,mse_val,mse_test,r_train,r_test,tr.time(end),tr.epoch(end)];
%alg_index=alg_index+1; 
%close all

save(strcat('matrix_results_',int2str(num_of_neurons),'_',int2str(1),'.mat'),'results_matrix')
save(strcat('training_results_',int2str(num_of_neurons),'_',int2str(1),'.mat'),'training_results')


f = scatteredInterpolant(X1_test,X2_test,predictions_test')
[X_test,Y_test] = meshgrid(X1_linear_train, X2_linear_train);

%hold on
Z_test = f(X_test,Y_test);
figure
mesh(X_test,Y_test,Z_test);
title('test predictions')
saveas(gcf, 'img/surfaceplot_test predictions_gd.png')


% net.numInputs = 2;
% net = configure(net,X_new);
% net = train(net, X_new, T_training_seq);
% view(net)
%a = sim(net, X_new);


% net_gd = feedforwardnet(num_of_neurons,'traingd');
% %net_gd = feedforwardnet();
% 
% net_gd.numInputs = length(X_new);
% 
% training_algos = {'traingd', 'trainlm', 'trainbfg'};
% 
% 
% net_gd = train(net_gd,X_new,T_training_seq);
% view(net_gd)
% a = sim(net_gd, X_new);


