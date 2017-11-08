function [ ll ] = ll_logit( y, X, beta )
%LL_LOGIT loglikelihood for binary logit

pr = 1.0 ./ (1.0 + exp(-X*beta));

ll = sum(y .* log(pr) + (1-y) .* log(1-pr));


end

