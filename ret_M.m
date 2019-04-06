function M = ret_M(inv_sqrt_g1,inv_sqrt_g1g2,inv_sqrt_g3)
  M=[0          inv_sqrt_g1     0               0               0;
	inv_sqrt_g1	0               inv_sqrt_g1g2	0               0;
	0           inv_sqrt_g1g2   0               inv_sqrt_g1g2	0;
	0           0               inv_sqrt_g1g2   0               inv_sqrt_g3;
	0           0               0               inv_sqrt_g3     0];
end