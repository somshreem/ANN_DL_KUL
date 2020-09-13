T = [1 1; -1 -1; 1 -1]';
plot(T(1,:),T(2,:),'r*')
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');

net_hopfield = newhop(T);

%% one input
%random_input = {rands(2,1)};  
%[output,Pf,Af] = net_hopfield({1 50},{},random_input);

%record = [cell2mat(random_input) cell2mat(output)];
%start = cell2mat(random_input);
hold on

%plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:))

%% 25 inputs
network_iteration = 50
random_input_generator = 25
plot(0,0,'ko');
color = 'rgbmy';
for i=1:random_input_generator
   random_input = {rands(2,1)};
   [output,Pf,Af] = sim(net_hopfield,{1 network_iteration},{},random_input);
   record=[cell2mat(random_input) cell2mat(output)];
   start=cell2mat(random_input);
   plot(start(1,1),start(2,1),'kx',record(1,:),record(2,:),color(rem(i,5)+1))
end
