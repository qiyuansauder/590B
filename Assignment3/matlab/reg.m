function [ beta, sigma2 ] = reg(Y, X, beta, sigma2, beta0, B0, nu0, delta0)
%REG Bayes regression function
%   Detailed explanation goes here

n = size(Y, 1);
% draw sigma2 
nu = nu0 + n; 
delta = delta0 + 1.0/sigma2 * (Y - X*beta)' * (Y - X*beta);
sigma2 = 1.0 / gamrnd(nu/2, 2/delta);

B = inv(inv(B0) + X'*X / sigma2);
betabar = B * (inv(B0)*beta0 + X'*Y / sigma2);

beta = mvnrnd(betabar, B)';
end

