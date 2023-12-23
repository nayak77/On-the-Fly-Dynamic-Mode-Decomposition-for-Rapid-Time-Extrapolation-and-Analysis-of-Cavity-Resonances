function [e_dmd,Phi_xe,lambda_p,omega_p,snap_e,S1,sigma_opt1,r,A_tild,W,D,L,snap_e1] = mrDMD_simple_opt(t_start,t_lim,dT,dT_act,e_fld_aft,N_1,N_t,dt_int,nstk)
dT_fac=round(dT/dT_act);


snap_e_dash=e_fld_aft(t_start:dT_fac:t_lim,:);
snap_e=snap_e_dash';


snap_EB=snap_e;  %**********modification in EB code************

N_sn = size(snap_EB,2);
N_dim = size(snap_EB,1);

%*********** this is for stacked DMD case*************
% nstk -> stacks wanted
stk_int = 1; % amount of shift
snap_e1 = snap_EB(:,1:1:end-nstk*stk_int);
snap_e2 = snap_EB(:,2:1:end-nstk*stk_int+1);

for i = 1:nstk-1
    snap_e1 = [snap_e1;snap_EB(:,1+i*stk_int:1:end-nstk*stk_int+i*stk_int)];
    snap_e2 = [snap_e2;snap_EB(:,2+i*stk_int:1:end-nstk*stk_int+i*stk_int+1)];
end



% we have the two snapshot matrices. Now we can implement the DMD algorithm


[U1,S1,V1]=svd(snap_e1,'econ');
beta = size(snap_e1,2) / size(snap_e1,1);
sigma_opt1=optimal_SVHT_coef(beta, 0)*median(diag(S1));
omeg = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43;
sigma_arr=diag(S1);
sigma_opt=omeg*median(sigma_arr);

diag_s1=diag(S1);
md_cut0=find(diag_s1>sigma_opt1, 1, 'last' );

r=md_cut0;
U1r=U1(:,1:r);
S1r=S1(1:r,1:r);
V1r=V1(:,1:r);

A_tild=U1r'*snap_e2*V1r/S1r;
[W,D,L]=eig(A_tild);

Phi=snap_e2*V1r/S1r*W;      % DMD modes

%Phi=U1r*W;
%d=(Phi'*Phi)\Phi'*snap_e1(:,1);
lambda=diag(D);

omega=log(lambda)/(dT*dt_int);




lambda_p=lambda;
omega_p=omega;          % lets not do pairing!!
Phi_p=Phi;

se1=snap_e1(:,1);
b_e=Phi_p\se1;    %this b has exp terms multiplied in it
%b=b_e./exp(omega*(t_start)*dT_act*dt_int);
b=b_e./exp(omega_p*(t_start)*dT_act*dt_int);
%b=b_e;

%N_t=200;        % number of time steps till which we want DMD to predict solution
%N_t=1200;
time_dyna=zeros(r,N_t); % the time is from 1 to N_t*20

for ii=1:(N_t)
    %time_dyna(:,ii)=(b.*exp(omega*(ii)*dT_act*dt_int));
    time_dyna(:,ii)=(b.*exp(omega_p*(ii)*dT_act*dt_int));
end


%*********for stacked case***********************************
%N_EB=5*N_1+4*N_2+1;   %**********changes for EB************
N_EB=N_1;
%N_EB=N_1+N_2;   %**********changes for EB************
%Phi_xe=Phi(1:N_EB,:); %**********changes for EB************
Phi_xe=Phi_p(1:N_EB,:); %**********changes for EB************


e_dmd=Phi_xe*time_dyna;




end
