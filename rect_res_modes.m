function [TM_mnp] = rect_res_modes(a,b,c,maxm,maxn,maxp,eps_r,mu_r)
%Returns TM mode frequencies for closed rectangular cavity
% a-> x length ; b-> y length ; c-> z length
% M -> max value of m,n,p
% eps -> Relative permittivity of medium
% mu -> Relative permeability of medium

eps = eps_r * 8.854e-12;
mu = mu_r * 4*pi*10^-7;

sz = (maxm)*(maxn)*(maxp+1);  % for now maximum value of M is 5
TM_mnp = zeros(sz,4); % 1st column -> n, 2nd column -> p, 3rd column -> q, 4th column -> frequencies
idx = 0;
for m = 1:maxm
    for n = 1:maxn
        for p = 0:maxp
            idx = idx + 1;
            TM_mnp(idx,1) = m;
            TM_mnp(idx,2) = n;
            TM_mnp(idx,3) = p;

            f_mnp = (1/(2*sqrt(mu*eps)))*sqrt( (m/a)^2 + (n/b)^2 + (p/c)^2 );
            TM_mnp(idx,4) = f_mnp;
            
        end
    end
end
[~,I] = sort(TM_mnp(:,4));

TM_mnp(:,1) = TM_mnp(I,1);
TM_mnp(:,2) = TM_mnp(I,2);
TM_mnp(:,3) = TM_mnp(I,3);
TM_mnp(:,4) = TM_mnp(I,4);
end

