function [ v ] = randtn(n, mu, sigma, a, b)
%RANDTN Return a truncated normal distribution with n values;
%   Detailed explanation goes here

u = rand(n, 1); 
fa = normcdf(a, mu, sigma);
fb = normcdf(b, mu, sigma);

v = mu + sigma .* norminv(u.*(fb-fa) + fa, 0.0, 1.0);

end
