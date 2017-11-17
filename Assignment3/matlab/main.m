%-------------------------------------------------------------------------
% Main estimation algorithm for Chan, Wu and Xie (2011)
% Converted from original gsl code
%-------------------------------------------------------------------------

%% read in data
process = csvread('data/process.csv', 1, 1);
amount = csvread('data/amount.csv', 1, 0);
customer = csvread('data/customer.csv', 1, 1);

%% define variables
n = size(process, 1);
nn = size(amount, 1);
record = process;
XX = [ones(n,1) customer];
t = amount(:, 2);
z = amount(:, 3);
p = size(XX, 2);
k = 1;
q = 1; % parameter for the amount equation
m = 2; % parameters for the process; 
d = 0.0031; % discount rate, equals to yearly of 20%
ind = int16(amount(:, 1)); %?
%% mcmc related variables
niters = 50000;

G0 = zeros(p, m+q);
G0(1,1) = -5.0;
G0(1,2) = -4.0;
G0(1,3) = 4.0;
G = G0;
A0 = 1.0*eye(p, p);
nu0 = 4;
delta0 = 2;
mu0 = 4;
V0 = 0.1*eye(m+q, m+q);
V = V0;
beta0 = ones(k, 1);
B0 = 2*eye(k, k);
Sigma = 2*eye(m+q, m+q);
beta = zeros(k, 1);
sigma2 = 1.0;

Theta = zeros(n, m+q);
Theta_prop = zeros(n, m+q);
Thetabar = Theta;

loglik = zeros(n, 1); 
loglik_prop = zeros(n, 1);

ThetaM = zeros(n, m+q, niters);
betaM = zeros(k, niters);
sigma2M = zeros(niters, 1);
GM = zeros(p, m+q, niters);
SigmaM = zeros(m+q, m+q, niters);

%% iterations
for m = 1:niters
    
    % theta_sample using random-walk metropolis-hastings;  
    Theta_prop = mvnrnd(Theta, V);
    
    loglik = ll_fn(Theta, beta, sigma2, record, t, z, ind); 
    loglik_prop = ll_fn(Theta_prop, beta, sigma2, record, t, z, ind);
    
    al = loglik_prop - loglik + ...
        log(mvnpdf(Theta_prop, Thetabar, Sigma)) - log(mvnpdf(Theta, Thetabar, Sigma));
    
    al = min(1, exp(al));
    draw = rand(n, 1);
    
    Theta(draw < al,:) = Theta_prop(draw < al,:);
    accept = mean(draw<al);
    
    % multireg update for G and Sigma
    [G, Sigma] = multireg(Theta, XX, G0, A0, mu0, V0);
    
    Thetabar = XX * G; 
    % update V and tune it;
    V = inv(V0 + inv(Sigma));
    V = -1.19 * V / norminv(accept/2.0, 0.0, 1.0);
    V = 0.5*(V+V');
    
    % regression for beta and sigma2
    bb = Theta(ind,3);
    [beta, sigma2] = reg(log(z)-bb, log(t), beta, sigma2, beta0, B0, nu0, delta0);
    
    % storage; 
    ThetaM(:,:,m) = Theta;
    betaM(:,m) = beta;
    sigma2M(m) = sigma2;
    GM(:,:,m) = G;
    SigmaM(:,:,m) = Sigma;
    
    if mod(m, 1000) == 1
        disp('------------------------------------------------------------')
        disp(['This message is sent at time ', datestr(now,'HH:MM:SS.FFF')]);
        disp(['Now at iteration ', num2str(m)]);
        disp(['acceptance rate for theta is ', num2str(accept)]);
        disp(['log likelihood is ', num2str(sum(loglik(:)))]);
        disp('mean theta is: ');
        disp(mean(Thetabar));
        disp(['beta is ', num2str(beta(:))]);
        disp(['sigma2 is ', num2str(sigma2)]);
        disp('G is: ');
        disp(G);
        disp('Sigma is: ');
        disp(Sigma);
    end
end

save results;

