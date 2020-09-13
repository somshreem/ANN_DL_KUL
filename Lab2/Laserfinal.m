% Followed example from 'nnet/TimeSeriesForecastingUsingDeepLearningExample'

%load dataset
data = load('lasertrain.dat');
data = data.';


%plot laser intensity dataset
figure
plot(data)
xlabel("Discrete Time")
ylabel("Intensity")
title("Santa Fe Laser Intensity")

%partition training dataset and train on first 90% of sequence and test on
%last 10%
numTimeStepsTrain = floor(0.9*numel(data));

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);


%standardize data to have zero mean and unit vaariance for better fit and 
%prevent training from diverging

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

%To forcast future values specify the responses to be the training 
%sequences with values shifted by one time step. At each time step of the 
%input sequence, the LSTM network learns to predict the value of the next time step

XTrain = dataTrainStandardized(5:end-1);
YTrain = dataTrainStandardized(6:end);

%Define LSTM network architecture : LSTM regression network : LSTM layer
%has 200 hidden units

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

%add training options. We set gradient threshold to 1 to avoid gradients
%from exploding

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

%train LSTM

net = trainNetwork(XTrain,YTrain,layers,options);

% Forcast future time steps - predict and update state function
% for each prediction use the previous prediction as input to the function
% also standardise the test data with same parameters as done for training
% data

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(5:end-1);


% initialize the network state, then predict on the training data XTrain. 
% Make the first prediction using the last time step of the training 
% response YTrain(end). Loop over the remaining predictions and input the 
% previous prediction to predictAndUpdateState.

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

%Unstandardize the predictions using the parameters calculated earlier

YPred = sig*YPred + mu;

% Calculate the RMSE from the unstandardized predictions

YTest = dataTest(6:end);
rmse = sqrt(mean((YPred-YTest).^2))

%Plot the training time series with the forecasted values

figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Intensity")
ylabel("")
title("Forecast")
legend(["Observed" "Forecast"])

%Compare the forecasted values with the test data.

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("")
title("Forecast without Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Intensity")
ylabel("Error")
title("RMSE = " + rmse)

%update network state with observed values

net = resetState(net);
net = predictAndUpdateState(net,XTrain);

%predict each timestep using observed value from previous timestep

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

%unstandardise predictions
YPred = sig*YPred + mu;

%calculate rmse
rmse = sqrt(mean((YPred-YTest).^2))

%compare forcasted values with test data
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Intensity")
ylabel("Error")
title("RMSE = " + rmse)