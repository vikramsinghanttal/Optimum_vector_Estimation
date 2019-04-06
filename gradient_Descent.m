function [g,N,D] = gradient_Descent(mu,g,lambda,No_of_iterations)
    N=0;
    D=0;
    for iter = 1:No_of_iterations
        Nr = lambda*g(1)*g(2)-lambda*lambda*lambda*g(1)*g(1)*g(2)*g(3);
        Dr = lambda*g(1)*g(2)+lambda*lambda*lambda*g(1)*g(1)*g(2)*g(3);
        Ni = g(3) + g(1) - lambda*lambda*g(1)*g(1)*g(2) - lambda*lambda*g(1)*g(2)*g(3);
        Di = g(3) - g(1) + lambda*lambda*g(1)*g(1)*g(2) - lambda*lambda*g(1)*g(2)*g(3);

        N  = (Nr*Nr + Ni*Ni);
        D  = (Dr*Dr + Di*Di);

        %     error(iter)     = N/D;

        del_Nr_g1 = lambda*g(2) - 2*lambda*lambda*lambda*g(1)*g(2)*g(3);
        del_Ni_g1 = 1 - 2*lambda*lambda*g(1)*g(2)-lambda*lambda*g(2)*g(3);
        del_Dr_g1 = lambda*g(2) + 2*lambda*lambda*lambda*g(1)*g(2)*g(3);
        del_Di_g1 = -1 + 2*lambda*lambda*g(1)*g(2)-lambda*lambda*g(2)*g(3);


        del_Nr_g2 = lambda*g(1) - lambda*lambda*lambda*g(1)*g(1)*g(3);
        del_Ni_g2 = -lambda*lambda*g(1)*g(1)-lambda*lambda*g(1)*g(3);
        del_Dr_g2 = lambda*g(1) + lambda*lambda*lambda*g(1)*g(1)*g(3);
        del_Di_g2 = lambda*lambda*g(1)*g(1)-lambda*lambda*g(1)*g(3);


        del_Nr_g3 = - lambda*lambda*lambda*g(1)*g(1)*g(2);
        del_Ni_g3 = 1-lambda*lambda*g(1)*g(2);
        del_Dr_g3 = lambda*lambda*lambda*g(1)*g(1)*g(2);
        del_Di_g3 = 1-lambda*lambda*g(1)*g(2);

        del_mse_g1 = 2*((Nr*del_Nr_g1+Ni*del_Ni_g1)*D-N*(Dr*del_Dr_g1+Di*del_Di_g1))/(D*D);
        del_mse_g2 = 2*((Nr*del_Nr_g2+Ni*del_Ni_g2)*D-N*(Dr*del_Dr_g2+Di*del_Di_g2))/(D*D);
        del_mse_g3 = 2*((Nr*del_Nr_g3+Ni*del_Ni_g3)*D-N*(Dr*del_Dr_g3+Di*del_Di_g3))/(D*D);

        grad_mse = [del_mse_g1;del_mse_g2;del_mse_g3];
        g = g - mu*grad_mse;
    end
end
    