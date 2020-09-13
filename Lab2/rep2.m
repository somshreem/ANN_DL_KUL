%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

T = [1 1; -1 -1; 1 -1]';


symmetric_initial_points = [0.2 0.2; 0.2 -0.2; -0.2 0.2; -0.2 -0.2; 0.1 0.1; 0.1 -0.1; -0.1 0.1; -0.1 -0.1]';

net = newhop(T);
n=8;
for i=1:n
    %a={rands(2,1)};                     % generate an initial point 
    input = {symmetric_initial_points(:,i)}
    
    [output,Pf,Af] = sim(net,{1 50},{},input);   % simulation of the network for 50 timesteps              
    record=[cell2mat(input) cell2mat(output)];   % formatting results  
    start=cell2mat(input);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');
