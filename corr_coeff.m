function [coeff] = corr_coeff(vec1,vec2)
% This function calculates the corr. coeff, given two vectors
% 
mean1=mean(vec1);
mean2=mean(vec2);

[sz1,~]=size(vec1);  % sz1 = sz2 (should be)
[sz2,~]=size(vec2);

self_corr1=0;
self_corr2=0;
cross_corr=0;

for i=1:sz1
    %self_corr1=self_corr1+(vec1(i)-mean1)^2;
    self_corr1=self_corr1+(vec1(i)-mean1).*conj(vec1(i)-mean1);
    %self_corr2=self_corr2+(vec2(i)-mean2)^2;
    self_corr2=self_corr2+(vec2(i)-mean2)*conj(vec2(i)-mean2);
    %cross_corr=cross_corr+(vec1(i)-mean1)*(vec2(i)-mean2);
    cross_corr=cross_corr+(vec1(i)-mean1)*conj(vec2(i)-mean2);
end

coeff=cross_corr/sqrt(self_corr1*self_corr2);
%coeff=cross_corr/sqrt(self_corr1*conj(self_corr2));


end

