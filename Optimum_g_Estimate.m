clear all;
clc;
%% Filter Parameters %%
n  = 3;                          % order of the filter
QuV= [inf 1000];                 % quality factor for lossless

%% System Parameters
f = linspace(0.5e9,1.5e9,200);  % range of frequency 0.5GHz to 1.5GHz
f_length =  length(f);           % Length of Frequency vector
f0 = 10^9;                       % resonant frequency 1GHz
BW = 10^8;                       % f2-f1

%% Gradient Descent Algorithm Parameters %%
mu  = 0.001*ones(f_length,1);
mu(90:110)  = 0.0001*ones(21,1);
No_of_iterations = 10000*ones(f_length,1);
No_of_iterations(90:110) = 10*ones(21,1);
%% g values
g1=1.5963;                       
g2=1.0967;
g3=1.5963;

g_opt_Qu = zeros(3,1);           % optimum value of g over whole frequency
                                 % band at quality factor Qu
g_opt_inf= zeros(3,1);           % optimum value of g over whole frequency
                                 % band at quality factor Qu=inf
g_inf_f = zeros(3,f_length);     % frequency specific best estimate of g at quality factor Qu 
g_Qu_f  = zeros(3,f_length);     % frequency specific best estimate of g at quality factor Qu = inf

%% R,I and M matrix
R= [1 0 0 0 0;
    0 0 0 0 0;
    0 0 0 0 0;
    0 0 0 0 0;
    0 0 0 0 1];
I=eye(n+2);
I(1,1)=0;
I(n+2,n+2)=0;

%% Sxx values
% S21=zeros(f_length,1);
S11_inf_opt_grad =	zeros(f_length,1);
S11_inf_g        =	zeros(f_length,1);
S11_inf_opt      =  zeros(f_length,1);
S11_Qu_opt_grad       =	zeros(f_length,1);
S11_Qu_g         =	zeros(f_length,1);
S11_Qu_opt       =  zeros(f_length,1);

%% For Qu = inf
Qu = QuV(1);

for fop=1:f_length
    fi = f(fop);
    
    lambda= (f0/BW)*((1/Qu)+(fi/f0)-(f0/fi));
    M=  ret_M(1/sqrt(g1),1/sqrt(g1*g2),1/sqrt(g3));       
    A= lambda*I - 1i*R + M;      
    
    inv_A = inv(A);
    S11_inf_g(fop)= 20*log10(abs(1+2*1i*inv_A(1,1)));
        
    g = [g1;g2;g3];             % Initial values for g (avoids local optima) 
    [g,N,D] = gradient_Descent(mu(fop),g,lambda,No_of_iterations(fop));
    g_inf_f(:,fop) = g;
    
    M=  ret_M(1/sqrt(g(1)),1/sqrt(g(1)*g(2)),1/sqrt(g(1)*g(2)));
    A= lambda*I - 1i*R + M;    
    inv_A = inv(A);
    S11_inf_opt(fop)= 20*log10(abs(1+2*1i*inv_A(1,1)));
    fop
end
figure
hold on
p1 = plot(f,S11_inf_g,'r-o');
p.LineWidth = 2;
p3 = plot(f,S11_inf_opt,'b:p');
hold on
%% For Qu = 1000

Qu = QuV(2);

for fop=1:f_length
    fi = f(fop);
    
    lambda= (f0/BW)*((1/Qu)+(fi/f0)-(f0/fi));
    M=  ret_M(1/sqrt(g1),1/sqrt(g1*g2),1/sqrt(g3));       
    A= lambda*I - 1i*R + M;      
    
    inv_A = inv(A);
    S11_Qu_g(fop)= 20*log10(abs(1+2*1i*inv_A(1,1)));
        
    g = [g1;g2;g3];             % Initial values for g (avoids local optima) 
    [g,N,D] = gradient_Descent(mu(fop),g,lambda,No_of_iterations(fop));
    g_Qu_f(:,fop) = g;
    
    M=  ret_M(1/sqrt(g(1)),1/sqrt(g(1)*g(2)),1/sqrt(g(1)*g(2)));
    A= lambda*I - 1i*R + M;    
    inv_A = inv(A);
    S11_Qu_opt(fop)= 20*log10(abs(1+2*1i*inv_A(1,1)));
    
    fop
end
hold on
q1 = plot(f,S11_Qu_g,'k-*');
q1.LineWidth = 2;
q3 = plot(f,S11_Qu_opt,'g:<');
legend('S11 : Using Aggregate Estimate [Qu=inf]','S11 : Frequency specific Estimate [Qu=inf]','S11 : Using Aggregate Estimate [Qu=1000]','S11 : Frequency specific Estimate [Qu=1000]');

