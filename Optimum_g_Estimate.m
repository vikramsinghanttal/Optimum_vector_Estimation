%%
% Issues : Newton Raphson can't be applied as gradient and Hessian(difficult to calculate)
%          doesn't exist at f=f0.
% Gradient Descent algorithm is applied which also diverges at f = f0 but
% can be made to perform well around f0 by configuring mu properly.
% For proper convergence rate mu needs to be set carefully.

% For analysis of S11 refer to analzeS11 can be found on https://github.com/vikramsinghanttal/Optimum_vector_Estimation
% For Linux system or git bash type following commands for download/cloneing this
% repository :
% git clone https://github.com/vikramsinghanttal/Optimum_vector_Estimation

% and for commiting and push the change type following
% git add filename.m
% git commit -m "describe commit comments"
% git push origin master // specify properly

% Thanks
% Vikram Singh (https://vikramsinghanttal.github.io/IIT-Kanpur/)

clear all;
clc;
%% Filter Parameters %%
n        = 3;                          % order of the filter
QuV      = [inf 1000];                 % quality factor for lossless

%% System Parameters
f        = linspace(0.5e9,1.5e9,200);  % range of frequency 0.5GHz to 1.5GHz
f_length =  length(f);           % Length of Frequency vector
f0       = 10^9;                       % resonant frequency 1GHz
BW       = 10^8;                       % f2-f1

%% Gradient Descent Algorithm Parameters %%
len_singularity = f_length/10;
low_singularity = f_length/2-len_singularity/2+1;
higher_singular = f_length/2+len_singularity/2;
mu              = 0.01*ones(f_length,1);                                            % Learning parameter or stepsize

mu(low_singularity:higher_singular)               = 0.0001*ones(len_singularity,1);
No_of_iterations                                  = 10000*ones(f_length,1);
No_of_iterations(low_singularity:higher_singular) = 10*ones(len_singularity,1);
% Value of mu has to be reduced as frequency goes towards f0(Resonance) to avoid the
% algorithm from diverging (because of Sxx's does exist at f = f0

%% g values
g1       = 1.5963;                       
g2       = 1.0967;
g3       = 1.5963;

g_opt_Qu = zeros(3,1);           % optimum value of g over whole frequency
                                 % band at quality factor Qu
g_opt_inf= zeros(3,1);           % optimum value of g over whole frequency
                                 % band at quality factor Qu=inf
g_inf_f  = zeros(3,f_length);     % frequency specific best estimate of g at quality factor Qu 
g_Qu_f   = zeros(3,f_length);     % frequency specific best estimate of g at quality factor Qu = inf

%% R and I matrix
R         =[1 0 0 0 0;
            0 0 0 0 0;
            0 0 0 0 0;
            0 0 0 0 0;
            0 0 0 0 1];
I         = eye(n+2);
I(1,1)    = 0;
I(n+2,n+2)= 0;

%% Sxx values
% S21=zeros(f_length,1);
S11_inf_g        =	zeros(f_length,1);  % S11 for Qu =inf, using g =[g1,g2,g3];
S11_inf_GD_f     =  zeros(f_length,1);  % S11 for Qu =inf, using g estimated from gradient descent[Freq specific];
S11_Qu_g_GD_agg  =	zeros(f_length,1);  % S11 for Qu =1000, using g estimated from gradient descent and aggregated(weighted) using signmoid function;
S11_Qu_g         =	zeros(f_length,1);  % S11 for Qu =1000, using g =[g1,g2,g3];
S11_Qu_GD_f      =  zeros(f_length,1);  % S11 for Qu =1000, using g estimated from gradient descent[Freq specific];

%% For Qu = inf
Qu = QuV(1);

for fop=1:f_length
    fi                = f(fop);
    
    lambda            = (f0/BW)*((1/Qu)+(fi/f0)-(f0/fi));
    M                 =  ret_M(1/sqrt(g1),1/sqrt(g1*g2),1/sqrt(g3));       
    A                 = lambda*I - 1i*R + M;      
    
    inv_A             = inv(A);
    S11_inf_g(fop)    = 20*log10(abs(1+2*1i*inv_A(1,1)));
        
    g                 = [g1;g2;g3];             % Initial values for g (avoids local minima) 
    [g,N,D]           = gradient_Descent(mu(fop),g,lambda,No_of_iterations(fop));
    g_inf_f(:,fop)    = g;         % Frequency optimum values of g
    
    M                 =  ret_M(1/sqrt(g(1)),1/sqrt(g(1)*g(2)),1/sqrt(g(1)*g(2)));
    A                 = lambda*I - 1i*R + M;    
    inv_A             = inv(A);
    S11_inf_GD_f(fop) = 20*log10(abs(1+2*1i*inv_A(1,1)));
    fop
end
%% Results for Qu = inf
figure
hold on
p1 = plot(f,S11_inf_g,'r-o');
p3 = plot(f,S11_inf_GD_f,'b:p');
p1.LineWidth = 2;
p3.LineWidth = 1.5;
hold on

%% For Qu = 1000
Qu      = QuV(2);

for fop=1:f_length
    fi              = f(fop);
    lambda          = (f0/BW)*((1/Qu)+(fi/f0)-(f0/fi));
    M               =  ret_M(1/sqrt(g1),1/sqrt(g1*g2),1/sqrt(g3)); 
    
    A               = lambda*I - 1i*R + M;      
    inv_A           = inv(A);
    S11_Qu_g(fop)   = 20*log10(abs(1+2*1i*inv_A(1,1)));
        
    g               = [g1;g2;g3];             % Initial values for g (avoids local minima)
    [g,N,D]         = gradient_Descent(mu(fop),g,lambda,No_of_iterations(fop));
    g_Qu_f(:,fop)   = g;
    
    M               =  ret_M(1/sqrt(g(1)),1/sqrt(g(1)*g(2)),1/sqrt(g(1)*g(2)));
    A               = lambda*I - 1i*R + M;    
    inv_A           = inv(A);
    S11_Qu_GD_f(fop)= 20*log10(abs(1+2*1i*inv_A(1,1)));
    
    fop
end

%% Sigmoid weighing of all g vectors
f_tau               = (f(f_length)-f(1))/10;                            % drop rate
sigmoid_prior       = exp(-abs(f-f0)./f_tau);                           % sigmoid weights calculation 
g_Qu                = sum(sigmoid_prior.*g_Qu_f,2)/(sum(sigmoid_prior));% aggregate vector


%% Calculation of S11 for aggregate vector g_Qu
for fop=1:f_length
    fi              = f(fop); 
    lambda          = (f0/BW)*((1/Qu)+(fi/f0)-(f0/fi));
    M               =  ret_M(1/sqrt(g_Qu(1)),1/sqrt(g_Qu(1)*g_Qu(2)),1/sqrt(g_Qu(3)));       
    A               = lambda*I - 1i*R + M;      
    inv_A           = inv(A);
    
    S11_Qu_g_GD_agg(fop)= 20*log10(abs(1+2*1i*inv_A(1,1)));
    fop
end


%% Results for Qu = 1000
hold on
q1           = plot(f,S11_Qu_g,'k-*');
q2           = plot(f,S11_Qu_g_GD_agg,'c:>');
q3           = plot(f,S11_Qu_GD_f,'g:<');
q1.LineWidth = 2;
q2.LineWidth = 1.5;
q3.LineWidth = 1.5;
legend(['S11 : Using Aggregate Estimate [Qu=' num2str(QuV(1)) ']'],['S11 : Frequency specific Estimate [Qu' num2str(QuV(1)) ']'],['S11 : Using Aggregate Estimate [Qu=' num2str(QuV(2)) ']'],['S11 : Using Aggregate Estimate [Qu=' num2str(QuV(2)) ']'],['S11 : Frequency specific Estimate [Qu=' num2str(QuV(2)) ']']);
ylabel('S_{11}(dB)');
xlabel('Frequency');
title('S_{11} vs Frequency for different Q_u and g coefficients (f_0 = 1 GHz)')