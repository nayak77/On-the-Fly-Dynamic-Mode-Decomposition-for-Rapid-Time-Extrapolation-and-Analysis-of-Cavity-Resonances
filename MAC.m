function [coeff] = MAC(vec1,vec2)
% This function calculates the MAC coeff, given two vectors

coeff=(vec1.'*conj(vec2))^2/( (vec1.'*conj(vec1))*(vec2.'*conj(vec2)) );



end

