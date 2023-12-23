function [pulse] = gpulse(tindex,dt,fcent,tshift,pwidth)
% tindex -> Discrete time index (array)
% dt -> Time step
% fcent -> Center frequency
% tshift -> Time shift (in terms of discrete time index)
% pwidth -> Pulse width in time

t_half = pwidth/2; % pulse width = 5 ns
n_half = round(t_half/dt);
sigma = n_half;
% sigsqr = 93025 for 5 ns, 3721 for 1 ns, 33489 for 3 ns; 
% Shift: 305 -> half width for 5 ns, 61 -> half width for 1 ns, 183 -> half width for 3 ns
%tau = 3*3377 + tshift;
%tau = 3*675 + tshift;
tau = 3*183 + tshift;
%sigsqr =  11404129;
sigsqr = 33489;
garg = -((tindex-tau).^2)./sigsqr;
sarg = tindex.*(2*pi*fcent*dt);
pulse = sin(sarg).*exp(garg);

end

