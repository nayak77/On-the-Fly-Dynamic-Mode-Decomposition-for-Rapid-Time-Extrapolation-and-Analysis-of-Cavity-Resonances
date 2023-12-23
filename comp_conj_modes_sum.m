function [sum_modes,sum_omegas,sum_lambdas,sum_bs,index_pairs] = comp_conj_modes_sum(modes,omegas,lambdas,bs)
% This fuction takes the complex conjugate (or real) modes, lambdas etc as 
% input and gives the sum of complex conjugate pairs.


Phi_xe_var=modes;
omega_var=omegas;       % defining the inputs
lambda_var=lambdas;
b_var=bs;
%                      formatSpec = ['dafaqs going on?\n'];    % for debugging purpose
%                      fprintf(formatSpec);

[ne_var,nm_var]=size(Phi_xe_var);

Phi_var_dumm=zeros(ne_var,2,1);  % base for 3D matrix ne*2*nm (ne*2 represents 2 comp-conj modes, one is zero
                         % for real mode case. and total 'nm' mode pairs).
omg_var_dumm=zeros(2,1);
lambda_var_dumm=zeros(2,1);
b_var_dumm=zeros(2,1); 
n=0;
mark_vec=zeros(nm_var,1);
index_pairs = zeros(2,1);

for i=1:nm_var
    r_flag=0;  % tracks whether the mode has comp conj pair or not
    c_flag=0;
    for j=i:nm_var

%        if(lambda_var(j,1)==conj(lambda_var(i,1)))
        if(abs(real(lambda_var(j,1))-real(lambda_var(i,1)))<10^-10 && abs(imag(lambda_var(j,1))+imag(lambda_var(i,1)))<10^-10)
            c_flag=1;
%             formatSpec = ['i=' num2str(i) ', j=' num2str(j) ', lambda_j=' num2str(lambda_var(j,1)) ', lambda_i=' num2str(conj(lambda_var(i,1))) '\n'];    % for debugging purpose
%                      fprintf(formatSpec);

           % if (Phi_xe_var(:,j)==conj(Phi_xe_var(:,i)))
                 r_flag=1;
                 mark_vec(i,1)=1;
                 mark_vec(j,1)=1;

                 if(i==j) % means the mode is real, no pair
                     mod_pair=[Phi_xe_var(:,i),zeros(ne_var,1)];
                     omg_pair=[omega_var(i,1);0];
                     lambda_pair=[lambda_var(i,1),0];
                     b_pair=[b_var(i,1);0];
%                      formatSpec = ['i=' num2str(i) ', j=' num2str(j) '\n'];    % for debugging purpose
%                      fprintf(formatSpec);
                 else
%                      formatSpec = ['i=' num2str(i) ', j=' num2str(j) '\n'];    % for debugging purpose
%                      fprintf(formatSpec);
                     mod_pair=[Phi_xe_var(:,i),Phi_xe_var(:,j)];
                     omg_pair=[omega_var(i,1);omega_var(j,1)];
                     lambda_pair=[lambda_var(i,1),lambda_var(j,1)];
                     b_pair=[b_var(i,1);b_var(j,1)];
                 end
                 if(i == 1)
                    index_pairs(1,1) = i;
                    index_pairs(2,1) = j;
                 else
                    index_pairs = [index_pairs,[i;j]];
                 end
                 
            n=n+1;
            Phi_var_dumm(:,:,n)=mod_pair;
            omg_var_dumm(:,n)=omg_pair;
            lambda_var_dumm(:,n)=lambda_pair;
            b_var_dumm(:,n)=b_pair;
%             else
%                 flag=-99;
%                 i
%                 formatSpec = '\n Modes are not comp-conj, though omegas are!!';    % for debugging purpose
%                 fprintf(formatSpec);
%             end           
           
        end
    end
    if((c_flag==0)&&(mark_vec(i,1)==0))  % contains only one element of complex conjugate pair
            formatSpec = 'Error: All modes are not complex conjugates or real!!';    % for debugging purpose
            fprintf(formatSpec);        
             mod_pair=[Phi_xe_var(:,i),conj(Phi_xe_var(:,i))];
             omg_pair=[omega_var(i,1);conj(omega_var(i,1))];
             lambda_pair=[lambda_var(i,1),conj(lambda_var(i,1))];
             b_pair=[b_var(i,1);conj(b_var(i,1))];
              n=n+1;
            Phi_var_dumm(:,:,n)=mod_pair;
            omg_var_dumm(:,n)=omg_pair;
            lambda_var_dumm(:,n)=lambda_pair;
            b_var_dumm(:,n)=b_pair;
    end
    if(r_flag==-99)   
        break;        
     end
end
     sum_modes = Phi_var_dumm;
     sum_omegas = omg_var_dumm;
     sum_lambdas = lambda_var_dumm;
     sum_bs = b_var_dumm;

end

