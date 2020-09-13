P=[2 1 -2 -1; 
   2 -2 2 1];
T=[0 1 0 1];
TF = 'hardlim';
LF = 'learnp'

net = newp(P,T,TF,LF);
net = init(net);

net.trainParam.epochs = 20;
[net,tr_descr] = train(net,P,T);

Pnew = [1;-0.3]
sim(net,Pnew)

%% demo1
X = [ -0.5 -0.5 +0.3 -0.1;  ...
      -0.5 +0.5 -0.5 +1.0];
T = [1 1 0 0];

net1_x_less_point_four = newp(X,T,TF,LF); % x cood is less than -0.4
net1_x_less_point_four = init(net1_x_less_point_four);
net1_x_less_point_four.trainParam.epochs = 20;
[net1_x_less_point_four,tr_descr] = train(net1_x_less_point_four,X,T);

plotpv(X,T);
plotpc(net1_x_less_point_four.IW{1},net1_x_less_point_four.b{1});

XX = repmat(con2seq(X),1,3);
TT = repmat(con2seq(T),1,3);
net_repeated_seq = adapt(net1_x_less_point_four,XX,TT);
plotpc(net_repeated_seq.IW{1},net_repeated_seq.b{1});

x = [0.7; 1.2];
y = net1_x_less_point_four(x);
plotpv(x,y);
point = findobj(gca,'type','line');
point.Color = 'red';

hold on;
plotpv(X,T);
plotpc(net1_x_less_point_four.IW{1},net1_x_less_point_four.b{1});
hold off;
