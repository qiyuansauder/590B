function [ loglik ] = ll_fn(Theta, beta, sigma2, record, t, z, ind)
%LL_FN computes the logliklihood function


x = record(:,1) - 1.0;
tx = record(:,2);
T = record(:,3);

mu = exp(Theta(:,1));
lambda = exp(Theta(:,2));
b = Theta(:,3);

loglik = x .* log(lambda) - log(lambda + mu) +...
    log(mu .* exp(-(lambda+mu) .* tx) + lambda .* exp(-(lambda+mu) .* T));

bb = b(ind);
loglik_z = log(normpdf(log(z), bb + beta .* log(t), sigma2));

loglik = loglik + accumarray(ind, loglik_z);
end

