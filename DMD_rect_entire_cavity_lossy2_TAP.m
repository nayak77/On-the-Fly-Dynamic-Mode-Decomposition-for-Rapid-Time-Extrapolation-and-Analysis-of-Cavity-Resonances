clc; clear all; close all;
% IEEE TAP journal: DMD code for the closed rectangular cavity with lossy dielectric
% Execute the sections sequentially

% load data
load init_data_rect_closed_lossy_f2w3.mat


%% Cavity Dimensions
dt_int = 8.1847045546213234E-012;
cavXleft = 20;
cavXright = 61;
cavYleft = 20;
cavYright = 101;
%cavYright = 116; 
cavZtop = 36;
%cavZtop = 44;
cavZbot = 15; 

x_arr = cavXleft+1:2:cavXright;
N_x = size(x_arr,2);
y_arr = cavYleft+1:2:cavYright;
N_y = size(y_arr,2);
z_arr = cavZbot+1:2:cavZtop-1;
N_z = size(z_arr,2);

% Preparing data
dx = 0.005;
dy = 0.005;
dz = 0.005;

cavXlength = (cavXright-cavXleft-1)*dx;
cavYlength = (cavYright-cavYleft-1)*dy;
cavZlength = (cavZtop-cavZbot)*dz; % staggered
%
ezRecAll_3D = reshape(ezReceiverAll(1,:),N_z,N_y,N_x,[]);
ezRecAll = reshape(ezReceiverAll(1,:),N_z*N_y*N_x,[]); % data for DMD
ezRecAll_3D = permute(ezRecAll_3D, [3,2,1,4]); % in the x,y,z,t format

x = dx*x_arr;
y = dy*y_arr; 
z = dz*(z_arr+0.5); % staggered
[X,Y,Z] = ndgrid(x,y,z);

X_min = min(min(min(X)));
X_max = max(max(max(X)));
Y_min = min(min(min(Y)));
Y_max = max(max(max(Y)));
Z_min = min(min(min(Z)));
Z_max = max(max(max(Z)));
%% Snapshot

% Number of points for plotting in x,y and z direction (proportional to the
% number of points in x,y and z direction.
x_plot_pnts = 84;
y_plot_pnts = 164; 
z_plot_pnts = 40;

xnew = linspace(X_min,X_max,x_plot_pnts);
ynew = linspace(Y_min,Y_max,y_plot_pnts); % make sure it's proportional to the length of each side
znew = linspace(Z_min,Z_max,z_plot_pnts);
[Xnew,Ynew,Znew] = meshgrid(xnew,ynew,znew);

tsn = 2001; % time snapshot for plotting

ezRecAll_3D_snap = ezRecAll_3D(:,:,:,tsn);

ezRecAll_3D_snapi = griddata(X,Y,Z,ezRecAll_3D_snap,Xnew,Ynew,Znew,'natural');
%
z_cut = 40;
y_cut = 80;
x_cut = 40;

xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,ezRecAll_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

xslice = X_max;
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,ezRecAll_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

%
xslice = X_min;
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezRecAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on


xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezRecAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

%
zslice = znew(z_cut); 
yslice = [];
xslice = [];
h1=slice(Xnew(y_cut:end,:,:),Ynew(y_cut:end,:,:),Znew(y_cut:end,:,:),ezRecAll_3D_snapi(y_cut:end,:,:),xslice,yslice,zslice);
h2=slice(Xnew(1:y_cut,x_cut:end,:),Ynew(1:y_cut,x_cut:end,:),Znew(1:y_cut,x_cut:end,:),ezRecAll_3D_snapi(1:y_cut,x_cut:end,:),xslice,yslice,zslice);
set(h1,'edgecolor','none')
set(h2,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on
%

zslice = []; 
yslice = ynew(y_cut);
xslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezRecAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on
%
zslice = []; 
yslice = Y_min;
xslice = [];
h1=slice(Xnew(1:y_cut,x_cut:end,1:z_cut),Ynew(1:y_cut,x_cut:end,1:z_cut),Znew(1:y_cut,x_cut:end,1:z_cut),ezRecAll_3D_snapi(1:y_cut,x_cut:end,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

colorTitleHandle = get(c,'Title');
titleString = 'V/m';
set(colorTitleHandle ,'String',titleString);
%
%
width = 2;
% Plot straight boundaries
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_min Z_min],'k-','LineWidth',width);
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_min],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[Y_max Y_max],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_min Z_min],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_min X_max],[Y_max Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min Y_min],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);

set(gca,'fontsize',24,'FontName','Times New Roman');


%% Initial data loading

y_hu_data0=ezRecAll(:,:);
y_hu_data_clean=y_hu_data0';
y_hu_data = y_hu_data_clean;
%
% Add noise
snr = 50;

%*********** if want to add noise, uncomment line below **********
%y_hu_data = awgn(y_hu_data_clean,snr);

% check by plotting
% figure;
% plot(y_hu_data_clean(10000:11000,4799));
% hold on
% plot(y_hu_data(10000:11000,4799));


[N_t,N_init]=size(y_hu_data);
t_final=N_t;
dT_act=2; % sampling rate in data aquired
dt_int = 8.1847045546213234E-012;

%% Selecting hyperparameters such as dT, win_width etc. for sliding-window DMD

%------------ selecting appropriate window width -------------
% Cavity dimensions
l = cavYlength;
wid = cavXlength;
h = cavZlength;
max_dim = sqrt(l^2+wid^2+h^2); % maximum dimension

% Determining the DMD window width
eps_r = 2.5;
mu_r = 1;
rec_modes = rect_res_modes(max_dim,max_dim,max_dim,2,2,2,eps_r,mu_r); % analytical resonance freqs for rectangular cavities
ref_freq = rec_modes(1,4); % fundamental mode frequency
ref_T = 1/ref_freq;
n_cycles = 30; % generally 30 cycles
win_width_T = n_cycles*ref_T; 
win_width = ceil(win_width_T/(dt_int*dT_act)); % window width in terms of time samples

% Fix start and end point in time over which FFT will be performed
t_start_init = 1;
t_end_init = t_start_init + win_width; 


% Parameters for FFT
dT_actr = 2;
num_pts = 20; % number of points (time series data) we want for FFT
rng(10)
s = rng;
qsize = numel(y_hu_data0(:,1));
qry_pts = randperm(qsize, num_pts);

% Preparing the time-series data for FFT
rec_arr = y_hu_data0(qry_pts,t_start_init:t_end_init); % only inside observed window
if(mod(win_width,2) == 0)
    nr = win_width-1;
else
    nr = win_width;
end
y_rec = rec_arr(:,1:nr);

% Perform FFT on all the time-series data and take average
f_rec = 0;
for i = 1:num_pts
    f1_rec = abs(fft(y_rec(i,:))).^1/nr;
    f_rec = f_rec + fftshift(f1_rec);
end
f_rec = f_rec./num_pts;
fsr = 1/(dT_actr*dt_int);
f_axisr = (-(nr-1)/2:(nr-1)/2)*(fsr/nr);

% Plot the average FFT
figure;
plot(10^-9*f_axisr,f_rec,'Linewidth',1.5);
xlim([0 5])
grid on
set(gca,'fontsize',20,'FontName','Times New Roman');
xlabel('Frequency (GHz)');
ylabel('FFT');
pbaspect([3 1 1])
hold on
n_rhalf = round((length(f_rec)+1)/2);
f_axisr_right = 10^-9*f_axisr(n_rhalf:end);
frec_max = max(f_rec);
%[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend','NPeaks',20);  % top 20 peaks
[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend','MinPeakHeight',0.01*frec_max); % find peaks more than 1% of heighst peak

max_freq = max(fres_rec)*10^9;
dmd_samp_freq = 2*max_freq;
dmd_samp_interval = 1/dmd_samp_freq;

dT = fix(dmd_samp_interval/dt_int);
dT_fac=round(dT/dT_act);
dT = dT_act * dT_fac;
%% Sliding-window DMD method ...
tic;
% Other Hyperparameters for the sliding-window method
chk_pts = 2; % number of points to check consecutively the convergence criterion
err_thr = 0.1; % 10% error threshold, the convergence criterion
nshft = 20;

win_shift = ceil(win_width/nshft); % amount of window shift
nstk = 20; % Number of Hankel stacks
err_arr = zeros(1000,1); % preallocate to a large value
N_qry = 25000; % maximum time-step upto which we want to query
%
t_start_init = 1; % starting point of sliding window method
win = 1;
while(1)
    % window boundaries
    t_start = t_start_init + win_shift*(win-1); 
    t_end = t_start + win_width;
    

    % DMD 
    [y_dmd1,Phi_xe_loc,lambda_loc,omega_loc,snap1_loc,S1,sigma_opt,r_opt,~,~,~,~,~] = mrDMD_simple_opt(t_start,t_end,dT,dT_act,y_hu_data,N_init,N_t,dt_int,nstk);
    [mod_num_loc,~]=size(lambda_loc);

    snap_loc=snap1_loc;
    pp=0;
    szlim=r_opt;

    sz_loc=mod_num_loc;
    [~,szt_loc]=size(snap_loc);
    vand_loc=zeros(sz_loc,szt_loc);
    for dd=1:szt_loc
        for cc=1:sz_loc
            vand_loc(cc,dd)=(lambda_loc(cc,1))^(dd);
        end
    end
    P_loc=(Phi_xe_loc'*Phi_xe_loc).*conj(vand_loc*vand_loc');

    q_loc=conj(diag(vand_loc*snap_loc'*Phi_xe_loc));

    b_e_loc=P_loc\q_loc;
    b_loc=b_e_loc;
    
    % Post-processing of DMD eigenvalues
    for i = 1:mod_num_loc
        absv = abs(lambda_loc(i,1));
        if(absv > 1)
            theta = angle(lambda_loc(i,1));
            lambda_loc(i,1) = exp(1i*theta);
        end    
    end
    omega_loc=log(lambda_loc)/(dT*dt_int);  
    
    time_dyna1_loc=zeros(r_opt,win_width+1);
    for ii=N_qry:N_qry + win_width
         time_dyna1_loc(:,ii-N_qry+1)=1.*b_loc.*exp(omega_loc*(ii-t_start+dT_fac)*dT_act*dt_int); % Note: here we use dT_act
    end
    
  
    data_recon=Phi_xe_loc*time_dyna1_loc;

    if(win>1)
        err = 0;
        den = 0;
        for j = 1:win_width+1
            err = err + norm(data_recon(:,j) - data_recon_prev(:,j));
            den = den + norm(data_recon_prev(:,j));
        end

        err_arr(win-1) = err/den;

    end
    
%     if(win == 10)
%         break
%     end
%     if(win == 8)
%         data_recon_win1 = data_recon;
%     end
    data_recon_prev = data_recon;
    
    % break condition (if at least 5 windows have been travelled, and the
    % convergence criterion is satified)
    if (win>5 && all(err_arr(win-chk_pts:win-1)<err_thr))
        err_arr(win:end) = []; % delete rest of the zero elements
        break
    elseif (t_end > N_qry)
        fprintf("Failed to detect equiibrium before target time");
        break
    end
    win
    t_start % just for debugging purpose
    t_end
    win = win + 1;
    
end

toc;
%% Decide the sampling frequency again with the offline data
rec_arr = y_hu_data0(qry_pts,t_start:t_end); % only inside observed window
if(mod(win_width,2) == 0)
    nr = win_width-1;
else
    nr = win_width;
end
y_rec = rec_arr(:,1:nr);

f_rec = 0;
for i = 1:num_pts
    f1_rec = abs(fft(y_rec(i,:))).^1/nr;
    f_rec = f_rec + fftshift(f1_rec);
end
f_rec = f_rec./num_pts;
fsr = 1/(dT_actr*dt_int);
f_axisr = (-(nr-1)/2:(nr-1)/2)*(fsr/nr);

figure;
plot(10^-9*f_axisr,f_rec,'Linewidth',1.5);
xlim([0 5])
grid on
set(gca,'fontsize',20,'FontName','Times New Roman');
xlabel('Frequency (GHz)');
ylabel('FFT');
pbaspect([3 1 1])
hold on

n_rhalf = round((length(f_rec)+1)/2);
f_axisr_right = 10^-9*f_axisr(n_rhalf:end);
frec_max = max(f_rec);
%[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend','NPeaks',20);  % top 20 peaks
[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend','MinPeakHeight',0.01*frec_max); 
%[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend'); 

max_freq = max(fres_rec)*10^9;
min_freq = min(fres_rec)*10^9;
max_T = 1/min_freq;
n_cycles = 20; % generally 20 cycles
win_width_T2 = n_cycles*max_T; 
win_width2 = ceil(win_width_T2/(dt_int*dT_act));

dmd_samp_freq = 2*max_freq;
dmd_samp_interval = 1/dmd_samp_freq;

dT = fix(dmd_samp_interval/dt_int);
dT_fac=round(dT/dT_act);
dT = dT_act * dT_fac;
%%
 %========================== time stamp ===============
 tic;

% nstk = 6;
% t_start = 4000;
% t_end = 8000;
% dT = 10;
% dT_fac = 5;
% win_width = 4000;
% Offline DMD
[y_dmd1,Phi_xe_loc,lambda_loc,omega_loc,snap1_loc,S1,sigma_opt,r,~,~,~,~,~] = mrDMD_simple_opt(t_start,t_end,dT,dT_act,y_hu_data,N_init,N_t,dt_int,nstk);

r
figure;   
s1_max=max(diag(S1));
semilogy(diag(S1),'-ob','MarkerSize',6,'LineWidth',1)
grid on
set(gca,'fontsize',20,'FontName','Times New Roman');
xlabel('Index');
ylabel('Singular Value');
pbaspect([1 1 1])

[mod_num_loc,~]=size(lambda_loc);


%*********** from here Vandermonde matrix creation starts*******************
snap_loc=snap1_loc;
pp=0;
szlim=r;

sz_loc=mod_num_loc;
[~,szt_loc]=size(snap_loc);
vand_loc=zeros(sz_loc,szt_loc);
for dd=1:szt_loc
    for cc=1:sz_loc
        vand_loc(cc,dd)=(lambda_loc(cc,1))^(dd);
    end
end
P_loc=(Phi_xe_loc'*Phi_xe_loc).*conj(vand_loc*vand_loc');

q_loc=conj(diag(vand_loc*snap_loc'*Phi_xe_loc));

b_e_loc=P_loc\q_loc;
b_loc=b_e_loc;
%-----------------------optimal b calculation ends-------------------------

b_loc2=Phi_xe_loc\snap1_loc(:,1);

N_sp = N_init;
t_start_loc = t_start;
t_lim_loc = t_end;
win_wid = t_end - t_start +1;
[sum_modes_loc,sum_omegas_loc,sum_lambdas_loc,sum_bs_loc,index_pairs] = comp_conj_modes_sum(Phi_xe_loc,omega_loc,lambda_loc,b_loc);
 [~,n_rmdd]=size(sum_lambdas_loc);
 n_rmdd

  

% Mean energy calculation (inside window)
energy_mean_loc=zeros(n_rmdd,1);
mode_norm_loc0=zeros(win_wid,1);

 for jj=1:n_rmdd
    time_dyna_loc_dumm=zeros(2,win_wid);
for ii=t_start:t_end
    time_dyna_loc_dumm(:,ii-t_start+1)=(sum_bs_loc(:,jj).*exp(sum_omegas_loc(:,jj)*(ii-t_start_loc+dT_fac)*dT_act*dt_int)); % Note: here we use dT_act
end 
sum_modes_loc_var_dum=sum_modes_loc(:,:,jj)*time_dyna_loc_dumm;
for kk = 1:win_wid
    mode_norm_loc0(kk,1) = norm(sum_modes_loc_var_dum(:,kk));
end
energy_mean_loc(jj,1)=mean(mode_norm_loc0(1:dT_fac:end,1));
end 


energy_total=sum(energy_mean_loc.^2);
%energy_total=sum(energy_last_loc);

[e_sort_f,e_idx_f]=sort(energy_mean_loc.^2);
%[e_sort_f,e_idx_f]=sort(energy_last_loc);
e_sort=flip(e_sort_f);
e_idx=flip(e_idx_f);
energy_thr=0.95;
for jj=1:n_rmdd
    if(sum(e_sort(1:jj,1))/energy_total>=energy_thr)
        jj_90=jj;
        break
    end
end

    mdnum_thr=jj_90;

if (sum(e_sort(1:mdnum_thr,1))/energy_total<energy_thr)
    formatSpec = 'total energy of dominant modes less than 95% \n';
    sum(e_sort(1:mdnum_thr,1))/energy_total% for debugging purpose
end

% Post-processing of DMD eigenvalues
for i = 1:mod_num_loc
    absv = abs(lambda_loc(i,1));
    if(absv > 1)
        theta = angle(lambda_loc(i,1));
        lambda_loc(i,1) = exp(1i*theta);
    end    
end
omega_loc=log(lambda_loc)/(dT*dt_int);
% Reconstruction
time_dyna1_loc=zeros(r,N_t);
for ii=1:N_t
     time_dyna1_loc(:,ii)=1.*b_loc.*exp(omega_loc*(ii-t_start+dT_fac)*dT_act*dt_int); % Note: here we use dT_act
end
e_dmd1_loc=Phi_xe_loc*time_dyna1_loc;


% Correlation coeff. among DMD modes (according to energy rank)
corr_coeff_mat_loc=zeros(n_rmdd,n_rmdd);
MAC_mat_loc=zeros(n_rmdd,n_rmdd);
for i=1:n_rmdd
    eidx1=e_idx(i,1);
    for j=1:n_rmdd
        eidx2=e_idx(j,1);
        arr1=sum_modes_loc(:,1,eidx1);
        arr2=sum_modes_loc(:,1,eidx2);
    corr_coeff_mat_loc(i,j)=abs(corr_coeff(arr1,arr2));
    MAC_mat_loc(i,j)=abs(MAC(arr1,arr2));
%     if(i==j)
%         corr_coeff_mat_loc(i,j)=0;
%     end
    end
end

sum_freqs_loc = sum_omegas_loc./(2*pi);
 %------------- tracking based on dominant modes ends--------------------------------------
%

% New eigenvalue plot

e_sort_norm = e_sort./max(e_sort);
 cmap = autumn(256);
 cmap = flip(cmap);
 %e_sort_log_norm = log10(e_sort)./max(e_sort);

 %========================== time stamp ===============
 toc;

 figure;
 circle(0,0,1);
 hold on
    for jr=n_rmdd:-1:1
        eee=e_idx(jr,1);
        engy = e_sort_norm(jr);
        cmap_idx = round(255*engy)+1;
        rgb_arr = cmap(cmap_idx,:);
        %scatter(real(sum_lambdas_loc(1,eee)),imag(sum_lambdas_loc(1,eee)),40,[0 .447 .741],'filled');
        scatter(real(sum_lambdas_loc(1,eee)),imag(sum_lambdas_loc(1,eee)),80,rgb_arr,'filled','MarkerEdgeColor','k','LineWidth',1);
       % text(real(sum_lambdas_loc(1,jr)),imag(sum_lambdas_loc(1,jr))+0.02,[ 'mode ' num2str(jr)],'Color','black','FontSize',12);  
        hold on
        if(imag(sum_lambdas_loc(1,eee))~=0)
        %scatter(real(sum_lambdas_loc(2,eee)),imag(sum_lambdas_loc(2,eee)),40,[0 .447 .741],'filled');
        scatter(real(sum_lambdas_loc(2,eee)),imag(sum_lambdas_loc(2,eee)),80,rgb_arr,'filled','MarkerEdgeColor','k','LineWidth',1);
        end
      
    end
    %
  %  for jr=1:mdnum_thr
   for jr=1:n_rmdd
       eee=e_idx(jr,1);
       if(jr<=mdnum_thr)
        %scatter(real(sum_lambdas_loc(1,eee)),imag(sum_lambdas_loc(1,eee)),700,[0 0.5 0],'Linewidth',2);
        if(imag(sum_lambdas_loc(1,eee))~=0)
        %scatter(real(sum_lambdas_loc(2,eee)),imag(sum_lambdas_loc(2,eee)),700,[0 0.5 0],'Linewidth',2);
        end
        txt = texlabel('lambda4');
        %txt1 =text(real(sum_lambdas_loc(1,eee))+0.02,imag(sum_lambdas_loc(1,eee))+0.01,['\lambda_', num2str(jr)],'Color','black','FontSize',16,'Interpreter', 'latex');
        %text(real(sum_lambdas_loc(1,eee))+0.08,imag(sum_lambdas_loc(1,eee))-0.12, ("\lambda_" + jr),'Color','black','FontSize',20);
        %txt.Interpreter = 'latex';
       end       
      
    end



ax = gca;
ax.FontSize = 24;
xlabel('Re\{\lambda\}','FontSize',24)
ylabel('Im\{\lambda\}','FontSize',24)
colormap(cmap)
c = colorbar;
colorTitleHandle = get(c,'Title');
titleString = 'Normalized Energy';
set(colorTitleHandle,'String',titleString);

ylim([-1.1 1.1]);
%circle(0,0,1);
xlim([-1.1 1.25]);
%axis equal;
pbaspect([2.35 2.2 1])
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
%pbaspect([1 1 1])
%
 %---------------------------------------------------------------
 figure;
 circle(0,0,1);
    for jr=1:n_rmdd
        scatter(real(sum_lambdas_loc(1,jr)),imag(sum_lambdas_loc(1,jr)),40,[0 .447 .741],'filled');
       % text(real(sum_lambdas_loc(1,jr)),imag(sum_lambdas_loc(1,jr))+0.02,[ 'mode ' num2str(jr)],'Color','black','FontSize',12);  
        hold on
        if(imag(sum_lambdas_loc(1,jr))~=0)
        scatter(real(sum_lambdas_loc(2,jr)),imag(sum_lambdas_loc(2,jr)),40,[0 .447 .741],'filled');
        end
      
    end
    
  %  for jr=1:mdnum_thr
   for jr=1:n_rmdd
       eee=e_idx(jr,1);
       if(jr<=mdnum_thr)
        scatter(real(sum_lambdas_loc(1,eee)),imag(sum_lambdas_loc(1,eee)),200,[0 0.7 0],'Linewidth',2);
        if(imag(sum_lambdas_loc(1,eee))~=0)
        scatter(real(sum_lambdas_loc(2,eee)),imag(sum_lambdas_loc(2,eee)),200,[0 0.7 0],'Linewidth',2);
        end
        text(real(sum_lambdas_loc(1,eee))+0.01,imag(sum_lambdas_loc(1,eee))+0.01,['mode ' num2str(jr)],'Color','black','FontSize',16);

       end       
      
    end



ax = gca;
ax.FontSize = 24;
xlabel('Re(\lambda)','FontSize',24)
ylabel('Im(\lambda)','FontSize',24)
ylim([-1.1 1.1]);
circle(0,0,1);
xlim([-1.1 1.1]);
%axis equal;
pbaspect([1 1 1])
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
pbaspect([1 1 1])

figure;
imagesc(MAC_mat_loc),colorbar
set(gca,'YDir','normal')
caxis([0 1])
ax=gca;
ax.FontSize = 24;
xlabel('Mode index');ylabel('Mode index');
set(gca,'fontsize',24,'FontName','Times New Roman');
pbaspect([1 1 1])

%
% Time Series
%y_min = -3;
%y_max = 3;
t0 = t_start;
tf = t_final;
this = linspace(t0*dT_act, tf*dT_act, (tf-t0)+1)';
h = this(2) - this(1);
%
y_hu_dmd = real(e_dmd1_loc');
ezDMDAll_3D = reshape(y_hu_dmd',N_z,N_y,N_x,[]);
ezDMDAll_3D = permute(ezDMDAll_3D, [3,2,1,4]);
nx_qry = 8;
ny_qry = 20;
nz_qry = 6;
grd_idx_arr = [300,3,10];
% y_min = min(y_hu_data(t_start:end,grd_idx_arr(1)));
% y_max = max(y_hu_data(t_start:end,grd_idx_arr(1)));
y_min = min(ezRecAll_3D(nx_qry,ny_qry,nz_qry,:));
y_max = max(ezRecAll_3D(nx_qry,ny_qry,nz_qry,:));
%
figure;
%plot(this(1:end),y_hu_data_clean(t_start:end,grd_idx_arr(1)),'k','Linewidth',2)
plot(this(1:end),squeeze(ezRecAll_3D(nx_qry,ny_qry,nz_qry,t_start:end)),'k','Linewidth',1)
hold on
%plot(this(1:end),y_hu_dmd(t_start:end,grd_idx_arr(1)),'r-','Linewidth',1)
plot(this(1:end),squeeze(ezDMDAll_3D(nx_qry,ny_qry,nz_qry,t_start:end)),'r--','Linewidth',1)
s=patch([t0*dT_act t_end*dT_act t_end*dT_act t0*dT_act], [y_min y_min y_max y_max],[0 1 0]);
alpha(s,0.15)
xl=xlabel('Time-steps (n)');
%yl=ylabel('E_z (source plane) [V/m]');
yl=ylabel('$E_z$ [V/m]');
yl.Interpreter = 'latex';
%xl.Interpreter = 'latex';
%yl.Interpreter = 'latex';
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
ax=gca;
pbaspect([4 1 1])
legend('Original Data','DMD Reconstruction');
%xlim([10000 11000])
ylim([1.1*y_min,1.1*y_max]);
ax.FontSize = 24;


%%
% Error plot
y_hu_dmd = real(e_dmd1_loc');

t0 = t_start;
tf = t_final;
this = linspace(t0*dT_act, tf*dT_act, (tf-t0)+1)';

y_min = 10^-3;
y_max = 10^-1;
pred_steps = size(y_hu_dmd(t_start:end,:),1);
%
err_rel_e=zeros(pred_steps,1);

% denominator calculation
den1 = zeros(pred_steps,1);
for i = 1:pred_steps
den1(i)=norm(y_hu_data_clean(t_start+i-1,:));
end
den = mean(den1);

for i=1:pred_steps
%err_rel_e(i)=norm(y_hu_dmd(t_start+i-1,:)-y_hu_data(t_start+i-1,:))/norm(y_hu_data(t_start+i-1,:));
err_rel_e(i)=norm(y_hu_dmd(t_start+i-1,:)-y_hu_data_clean(t_start+i-1,:))/den;
end
hold off
%
y_min = 10^-6;
y_max = 10^-1;
figure;
semilogy(this,err_rel_e,'Linewidth',1.5)
hold on
s=patch([t0*dT_act t_end*dT_act t_end*dT_act t0*dT_act], [y_min y_min y_max y_max],[0 1 0]);
alpha(s,0.15)
xl=xlabel('Time-steps $(n)$');
yl=ylabel('$\delta^{(n)}$');
xl.Interpreter = 'latex';
yl.Interpreter = 'latex';
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
ylim([y_min y_max])
%% Plotting 3D DMD Modes (entire volume)

%fn_loc=real(sum_bs_loc(1,:).*sum_modes_loc(:,1,:)+sum_bs_loc(2,:).*sum_modes_loc(:,2,:));
fn_loc=real(sum_bs_loc(1,:).*sum_modes_loc(:,1,:));
[~,n_md]=size(sum_lambdas_loc);

org_fn_loc=zeros(N_sp,n_md);
for j=1:n_md
    org_fn_loc(:,j)=fn_loc(:,1,j);
end

DMDmodes_3D = reshape(org_fn_loc,N_z,N_y,N_x,[]);
%ezRecAll_3D = reshape(ezReceiverAll(1,:),N_z,N_phi,N_rho,[]);
%ezRecAll_3D(:,:,1,:) = ezRecAll_3D(:,:,1,:)*maxPhi; % normalizing fields on axis
%ezRecAll2 = reshape(ezRecAll_3D,N_z*N_phi*N_rho,[]); % data for DMD

DMDmodes_3D = permute(DMDmodes_3D, [3,2,1,4]); % in the x,y,z,t format
%
for md_idx =1:6 % number of modes to plot
DMDmodes_3D_snap = DMDmodes_3D(:,:,:,e_idx(md_idx));
%DMDmodes_3D_snap = DMDmodes_3D(:,:,:,md_idx);
%
DMDmodes_3D_snapi = griddata(X,Y,Z,DMDmodes_3D_snap,Xnew,Ynew,Znew,'natural');

z_cut = 40;
y_cut = 82; %60
x_cut = 42-21; % 60

figure;

xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,DMDmodes_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

xslice = X_max;
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,DMDmodes_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on


xslice = X_min;
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),DMDmodes_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),DMDmodes_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

zslice = znew(z_cut); 
yslice = [];
xslice = [];
h1=slice(Xnew(y_cut:end,:,:),Ynew(y_cut:end,:,:),Znew(y_cut:end,:,:),DMDmodes_3D_snapi(y_cut:end,:,:),xslice,yslice,zslice);
h2=slice(Xnew(1:y_cut,x_cut:end,:),Ynew(1:y_cut,x_cut:end,:),Znew(1:y_cut,x_cut:end,:),DMDmodes_3D_snapi(1:y_cut,x_cut:end,:),xslice,yslice,zslice);
set(h1,'edgecolor','none')
set(h2,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

zslice = []; 
yslice = Y_min;
xslice = [];
h1=slice(Xnew(1:y_cut,x_cut:end,1:z_cut),Ynew(1:y_cut,x_cut:end,1:z_cut),Znew(1:y_cut,x_cut:end,1:z_cut),DMDmodes_3D_snapi(1:y_cut,x_cut:end,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
%c=colorbar;
%c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on


zslice = []; 
yslice = ynew(y_cut);
xslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),DMDmodes_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
colorTitleHandle = get(c,'Title');
titleString = 'V/m';
set(colorTitleHandle ,'String',titleString);

hold on
%

width = 2;
% Plot straight boundaries
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_min Z_min],'k-','LineWidth',width);
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_min],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[Y_max Y_max],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_min Z_min],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_min X_max],[Y_max Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min Y_min],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);

title("Mode " + md_idx);
set(gca,'fontsize',24,'FontName','Times New Roman');

end
%% Source & Receiver FFT
dt_int = 8.1847045546213234E-012;

% Source ............
dT_acts = 1;  % Because source trace was collected at each time-step

tindex_f = length(receiverTrace);

tindex = linspace(0,tindex_f-1,tindex_f);
dt = dt_int;
fcent1 = 2*10^9;
fcent2 = 5*10^9;
tshift = 0;
pwidth1 = 5*10^-9; % 5 ns
pwidth2 = 1*10^-9; % 1 ns
pwidth3 = 300*10^-9; % 1 ns

pulse_f2w1 = 10^-5*gpulse(tindex,dt,fcent1,tshift,pwidth3);

sourceTrace = pulse_f2w1';

if(mod(size(sourceTrace,1),2) == 0)
    ns = size(sourceTrace,1)-1;
else
    ns = size(sourceTrace,1);
end
y_org = sourceTrace(1:ns,1);

f1_org = abs(fft(y_org)).^1/ns;
f_org = fftshift(f1_org);

fss = 1/(dT_acts*dt_int);
f_axiss = (-(ns-1)/2:(ns-1)/2)*(fss/ns);

% Receiver ............
%ezRecAll = reshape(ezReceiverAll(1,:),N_z*N_phi*N_rho,[]);
%y_hu_data0=ezRecAll(:,1:2:end);
dT_actr = 2;

% ************ We already have the "qry_pts" from sliding window *********
% ************ We just need to take large number of points **************

rec_arr = y_hu_data0(qry_pts,t_start:end);


if(mod(size(rec_arr,2),2) == 0)
    nr = size(rec_arr,2)-1;
else
    nr = size(rec_arr,2);
end
y_rec = rec_arr(:,1:nr);
% nr = t_end-t_start+1;
% y_rec = rec_arr(:,1:4001);

f_rec = 0;
for i = 1:num_pts
    f1_rec = abs(fft(y_rec(i,:))).^1/nr;
    f_rec = f_rec + fftshift(f1_rec);
end
f_rec = f_rec./num_pts;
fsr = 1/(dT_actr*dt_int);
f_axisr = (-(nr-1)/2:(nr-1)/2)*(fsr/nr);

% Analytical (closed rectangular cavity) .........

cavXlength = (cavXright-cavXleft-1)*dx;
cavYlength = (cavYright-cavYleft-1)*dy;
cavZlength = (cavZtop-cavZbot-1)*dz; % staggered

l = cavYlength  ;
wid = cavXlength ;
h = cavZlength ;

eps_r = 2.5;
mu_r = 1;
maxm = 12;
maxn = 12;
maxp = 4;
TM_mnp = rect_res_modes(wid,l,h,maxm,maxn,maxp,eps_r,mu_r);
res_f = 10^-9*TM_mnp(:,4);
nf = size(res_f,1);
maxval = 1.1*max(f_rec);


figure;
yyaxis left
p1 = plot(10^-9*f_axiss,f_org,'k','Linewidth',1.5);
xlim([0 5])
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
xlabel('Frequency (GHz)');
ylabel('FFT (source)');
pbaspect([2 1 1])

%
yyaxis right
p2 = plot(10^-9*f_axisr,f_rec,'Linewidth',1.5);
xlim([0 5])
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
xlabel('Frequency (GHz)');
ylabel('FFT (receiver)');
pbaspect([2 1 1])
hold on
ax = gca;
legend([p1,p2], 'Source','Receiver')
xlim([1.4 2.6])
ax.YAxis(1).Color = 'k';
%ax.YAxis(2).Color = 'k';
% for i = 1:nf
%     plot([res_f(i),res_f(i)],[0,maxval],'k-')
% end
%% Compare and pair analytical, DMD and FDTD frequencies

% First find peaks from FFT of FDTD solution (peaks of right half f_rec)
n_rhalf = round((length(f_rec)+1)/2);
f_axisr_right = 10^-9*f_axisr(n_rhalf:end);
frec_max = max(f_rec);
[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend','NPeaks',100);  % top 20 peaks

%[fpks_rec,fres_rec] = findpeaks(f_rec(n_rhalf:end),f_axisr_right,'SortStr','descend','MinPeakHeight',0.01*frec_max);  % top 20 peaks

% Analytical - DMD pairs: Use analytical freqs as reference
freqs_DMD =10^-9*abs(imag(sum_freqs_loc(1,:)))'; 
freqs_DMD_real = 10^-9*abs(real(sum_freqs_loc(1,:)))';

freqs_anal0 = 10^-9*TM_mnp(:,4); % reference in GHz
freqs_anal = freqs_anal0(freqs_anal0<6); % reference in GHz (< 6 GHz)
nfa = size(freqs_anal,1);

A = repmat(freqs_DMD,[1 length(freqs_anal)]);
[minValue,closestIndex_anal_DMD] = min(abs(A-freqs_anal'));
closestValue_anal_DMD = freqs_DMD(closestIndex_anal_DMD');
closestValue_anal_DMDr = freqs_DMD_real(closestIndex_anal_DMD');

anal_DMD_comp = [TM_mnp(1:nfa,1:3),freqs_anal,closestValue_anal_DMD,closestIndex_anal_DMD'];



% Analytical - FDTD pairs: Use Analytical freqs as reference
freqs_FDTD = fres_rec';
A = repmat(freqs_FDTD,[1 length(freqs_anal)]);
[minValue,closestIndex_anal_FDTD] = min(abs(A-freqs_anal'));
closestValue_anal_FDTD = freqs_FDTD(closestIndex_anal_FDTD');

ndmd = length(closestIndex_anal_DMD);
energy_idx = zeros(ndmd,1);
for i = 1:ndmd
    energy_idx(i,1) = find(e_idx == closestIndex_anal_DMD(i));
end
anal_FDTD_comp = [TM_mnp(1:nfa,1:3),freqs_anal,closestValue_anal_FDTD];

anal_FDTD_DMDi_comp = [anal_FDTD_comp,closestValue_anal_DMD,closestIndex_anal_DMD',energy_idx];

anal_FDTD_DMDr_comp = [anal_FDTD_comp,closestValue_anal_DMDr,closestIndex_anal_DMD',energy_idx];


% FDTD - analytical pairs: Use FDTD freqs as reference
freqs_FDTD = fres_rec';
A = repmat(freqs_anal,[1 length(freqs_FDTD)]);
[minValue,closestIndex_FDTD_anal] = min(abs(A-freqs_FDTD'));
closestValue_FDTD_anal = freqs_anal(closestIndex_FDTD_anal');
closestValue_FDTD_anal_idx = TM_mnp(closestIndex_FDTD_anal',1:3);
FDTD_anal_comp = [freqs_FDTD,closestValue_FDTD_anal];


% FDTD - DMD pairs: Use FDTD freqs as reference

A = repmat(freqs_DMD,[1 length(freqs_FDTD)]);
[minValue,closestIndex_FDTD_DMD] = min(abs(A-freqs_FDTD'));
closestValue_FDTD_DMD = freqs_DMD(closestIndex_FDTD_DMD');
closestValue_FDTD_DMDr = freqs_DMD_real(closestIndex_FDTD_DMD');

ndmd2 = length(closestIndex_FDTD_DMD);
energy_idx2 = zeros(ndmd2,1);
for i = 1:ndmd2
    energy_idx2(i,1) = find(e_idx == closestIndex_FDTD_DMD(i));
end

FDTD_DMD_comp = [freqs_FDTD,closestValue_FDTD_DMD,closestIndex_FDTD_DMD', energy_idx2];

FDTD_DMDi_anal_comp = [FDTD_DMD_comp,closestValue_FDTD_anal,closestValue_FDTD_anal_idx];
%% Calculate material parameter (sigma or imaginary part of permittivity) from exponentially decay term
m = 3;
n = 5;  %TM_mnp mode for which we want to calculate real and imaginary part of frequency
p = 0;
sigmaa = 0.0005;
eps0 = 8.854e-12;
% TM_mnp previously calculated
res_f = 10^-9*TM_mnp(:,4);
nf = size(anal_FDTD_DMDi_comp,1);
f_idx = 0;
for i = 1:nf
    if(anal_FDTD_DMDi_comp(i,1) == m && anal_FDTD_DMDi_comp(i,2) == n && anal_FDTD_DMDi_comp(i,3) == p)
        scal_fac = 10^9*anal_FDTD_DMDi_comp(i,4)*sqrt(eps_r);
        scal_fac_dmd = 10^9*anal_FDTD_DMDi_comp(i,6)*sqrt(eps_r);
        f_idx = i;
        break
    end
end

fcn = @(wr) wr - scal_fac*sqrt((sqrt(eps_r^2+(sigmaa/wr*eps0)^2)+eps_r)/2)/sqrt(eps_r^2+(sigmaa/wr*eps0)^2);
%wr_init = TM_mnp(f_idx,4);
wr_init = 10^9*anal_FDTD_DMDi_comp(f_idx,4);

wr_sol = fsolve(fcn,wr_init); % It appears that wr_sol = wr_init
% wr_init = 10^9*anal_FDTD_DMDi_comp(f_idx,4);
% wr_sol = wr_init;

epsr_imag = sigmaa/(2*pi*wr_sol*eps0);
epsr_real = eps_r;
epsr = epsr_real-1i*epsr_imag;

%fr_dmd = 10^9*FDTD_DMDr_anal_comp(f_idx,2);

w_cplx = (scal_fac/sqrt(epsr))
anal_FDTD_DMDr_comp(f_idx,6)*10^9
Q_org = real(w_cplx)/(2*abs(imag(w_cplx)))
Q_dmd = anal_FDTD_DMDi_comp(f_idx,6)*10^9/(2*anal_FDTD_DMDr_comp(f_idx,6)*10^9)
%% FFT comparison b/w FDTD and DMD

dT_actr = 2;

% Ground truth (FFT from long FDTD data)
fdtd_long = y_hu_data0(qry_pts,t_start:end);
if(mod(size(fdtd_long,2),2) == 0)
    n_long = size(fdtd_long,2)-1;
else
    n_long = size(fdtd_long,2);
end
fdtd_long = fdtd_long(:,1:n_long);
f_long = 0;
for i = 1:num_pts
    f1_long = abs(fft(fdtd_long(i,:))).^1/n_long;
    f_long = f_long + fftshift(f1_long);
end
f_long = f_long./num_pts;
fs_long = 1/(dT_actr*dt_int);
f_axis_long = (-(n_long-1)/2:(n_long-1)/2)*(fs_long/n_long);


% Coarse FFT (FFT from short FDTD data)
fdtd_short = y_hu_data0(qry_pts,t_start:t_end);
if(mod(size(fdtd_short,2),2) == 0)
    n_short = size(fdtd_short,2)-1;
else
    n_short = size(fdtd_short,2);
end
fdtd_short = fdtd_short(:,1:n_short);
f_short = 0;
for i = 1:num_pts
    f1_short = abs(fft(fdtd_short(i,:))).^1/n_short;
    f_short = f_short + fftshift(f1_short);
end
f_short = f_short./num_pts;
fs_short = 1/(dT_actr*dt_int);
f_axis_short = (-(n_short-1)/2:(n_short-1)/2)*(fs_short/n_short);

% DMD FFT (FFT from inter+extrapolated DMD data)
dmd_long = real(y_hu_dmd(t_start:end,qry_pts)');
if(mod(size(dmd_long,2),2) == 0)
    n_dmd = size(dmd_long,2)-1;
else
    n_dmd = size(dmd_long,2);
end
dmd_long = dmd_long(:,1:n_dmd);
f_dmd = 0;
for i = 1:num_pts
    f1_dmd = abs(fft(dmd_long(i,:))).^1/n_dmd;
    f_dmd = f_dmd + fftshift(f1_dmd);
end
f_dmd = f_dmd./num_pts;
fs_dmd = 1/(dT_actr*dt_int);
f_axis_dmd = (-(n_dmd-1)/2:(n_dmd-1)/2)*(fs_dmd/n_dmd);

figure;
plot(10^-9*f_axis_short,f_short,'Linewidth',1.5);
hold on
plot(10^-9*f_axis_long,f_long,'k','Linewidth',1.5);
plot(10^-9*f_axis_dmd,f_dmd,'r--','Linewidth',1.5);
legend('Short FFT','Ground truth','DMD-FFT');
xlim([1.5 2.5])
grid(gca,'minor')
grid on
set(gca,'fontsize',24,'FontName','Times New Roman');
xlabel('Frequency (GHz)');
ylabel('FFT');
pbaspect([2 1 1])
%% Snapshot comparison b/w FDTD and DMD


% Number of points for plotting in x,y and z direction (proportional to the
% number of points in x,y and z direction.
x_plot_pnts = 84;
y_plot_pnts = 164; 
z_plot_pnts = 40;

xnew = linspace(X_min,X_max,x_plot_pnts);
ynew = linspace(Y_min,Y_max,y_plot_pnts); % make sure it's proportional to the length of each side
znew = linspace(Z_min,Z_max,z_plot_pnts);
[Xnew,Ynew,Znew] = meshgrid(xnew,ynew,znew);

tsn = 24001; % time snapshot for plotting

ezRecAll_3D_snap = ezRecAll_3D(:,:,:,tsn);
ezDMDAll_3D_snap = ezDMDAll_3D(:,:,:,tsn);
ezRecAll_3D_snapi = griddata(X,Y,Z,ezRecAll_3D_snap,Xnew,Ynew,Znew,'natural');
ezDMDAll_3D_snapi = griddata(X,Y,Z,ezDMDAll_3D_snap,Xnew,Ynew,Znew,'natural');
%
cmax1 = max(max(max(ezRecAll_3D_snapi)));
cmax2 = max(max(max(ezDMDAll_3D_snapi)));
cmax = max(cmax1,cmax2);

cmin1 = min(min(min(ezRecAll_3D_snapi)));
cmin2 = min(min(min(ezDMDAll_3D_snapi)));
cmin = min(cmin1,cmin2);
%
z_cut = 40;
y_cut = 80;
x_cut = 60;

%subplot(1,2,1)
figure;
xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,ezRecAll_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

xslice = X_max;
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,ezRecAll_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

%
xslice = X_min;
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezRecAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on


xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezRecAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

%
zslice = znew(z_cut); 
yslice = [];
xslice = [];
h1=slice(Xnew(y_cut:end,:,:),Ynew(y_cut:end,:,:),Znew(y_cut:end,:,:),ezRecAll_3D_snapi(y_cut:end,:,:),xslice,yslice,zslice);
h2=slice(Xnew(1:y_cut,x_cut:end,:),Ynew(1:y_cut,x_cut:end,:),Znew(1:y_cut,x_cut:end,:),ezRecAll_3D_snapi(1:y_cut,x_cut:end,:),xslice,yslice,zslice);
set(h1,'edgecolor','none')
set(h2,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on
%

zslice = []; 
yslice = ynew(y_cut);
xslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezRecAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on
%
zslice = []; 
yslice = Y_min;
xslice = [];
h1=slice(Xnew(1:y_cut,x_cut:end,1:z_cut),Ynew(1:y_cut,x_cut:end,1:z_cut),Znew(1:y_cut,x_cut:end,1:z_cut),ezRecAll_3D_snapi(1:y_cut,x_cut:end,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

colorTitleHandle = get(c,'Title');
titleString = 'V/m';
set(colorTitleHandle ,'String',titleString);
%
%
width = 2;
% Plot straight boundaries
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_min Z_min],'k-','LineWidth',width);
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_min],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[Y_max Y_max],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_min Z_min],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_min X_max],[Y_max Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min Y_min],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);
caxis([cmin cmax])
set(gca,'fontsize',24,'FontName','Times New Roman');
title("FDTD");

%% DMD



%subplot(1,2,2)
figure;
xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,ezDMDAll_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

xslice = X_max;
zslice = [];   
yslice = [];
h1=slice(Xnew,Ynew,Znew,ezDMDAll_3D_snapi,xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

%
xslice = X_min;
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezDMDAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on


xslice = xnew(x_cut);
zslice = [];   
yslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezDMDAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

%
zslice = znew(z_cut); 
yslice = [];
xslice = [];
h1=slice(Xnew(y_cut:end,:,:),Ynew(y_cut:end,:,:),Znew(y_cut:end,:,:),ezDMDAll_3D_snapi(y_cut:end,:,:),xslice,yslice,zslice);
h2=slice(Xnew(1:y_cut,x_cut:end,:),Ynew(1:y_cut,x_cut:end,:),Znew(1:y_cut,x_cut:end,:),ezDMDAll_3D_snapi(1:y_cut,x_cut:end,:),xslice,yslice,zslice);
set(h1,'edgecolor','none')
set(h2,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on
%

zslice = []; 
yslice = ynew(y_cut);
xslice = [];
h1=slice(Xnew(y_cut:end,:,1:z_cut),Ynew(y_cut:end,:,1:z_cut),Znew(y_cut:end,:,1:z_cut),ezDMDAll_3D_snapi(y_cut:end,:,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on
%
zslice = []; 
yslice = Y_min;
xslice = [];
h1=slice(Xnew(1:y_cut,x_cut:end,1:z_cut),Ynew(1:y_cut,x_cut:end,1:z_cut),Znew(1:y_cut,x_cut:end,1:z_cut),ezDMDAll_3D_snapi(1:y_cut,x_cut:end,1:z_cut),xslice,yslice,zslice);
set(h1,'edgecolor','none')
colormap('jet');
c=colorbar;
c.FontSize = 24;
pbaspect([1 1 1])
axis equal
hold on

colorTitleHandle = get(c,'Title');
titleString = 'V/m';
set(colorTitleHandle ,'String',titleString);
%
%
width = 2;
% Plot straight boundaries
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_min Z_min],'k-','LineWidth',width);
plot3([xnew(x_cut) X_max],[Y_min Y_min],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_min],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_max X_max],[Y_min Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min X_min],[Y_max Y_max],[Z_min Z_max],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_min Z_min],'k-','LineWidth',width);
plot3([X_min X_min],[ynew(y_cut) Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([X_min X_max],[Y_max Y_max],[Z_max Z_max],'k-','LineWidth',width);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min Y_min],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_max],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);
plot3([X_min xnew(x_cut)],[ynew(y_cut) ynew(y_cut)],[Z_max Z_max],'k-.','LineWidth',width/2);
plot3([xnew(x_cut) xnew(x_cut)],[Y_min ynew(y_cut)],[Z_min Z_min],'k-.','LineWidth',width/2);
caxis([cmin cmax])
set(gca,'fontsize',24,'FontName','Times New Roman');
title("DMD");