
% --------------------------------------------------------------
% Demonstation of the binary probit model estimation using data
% augmentation
% --------------------------------------------------------------

%% simulate data
clear;
rng(314159);
n = 1000;
X = mvnrnd([1.5 -2.0], [1.0, 0.4; 0.4, 1.2], n);
X = [ones(n, 1), X];
beta_true = [-0.5; 2.1; 0.8];
e = randn(n,1);
y = X*beta_true + e >0;

%% priors and storage
niters = 10000;
k = size(X,2);
beta0 = zeros(k,1);
B0 = 10.0*eye(k);

beta = zeros(k, 1);
z = zeros(n, 1); % augmented data;

betaM = zeros(niters, k);

%% doing the MCMC loop 
for m = 1:niters
    % draw z based on y and beta
    
    % lower bound and upper bound
    Xbeta = X*beta;
    a = zeros(n, 1);
    b = zeros(n, 1);
    a(y==0) = -Inf;
    b(y==1) = Inf;
    
    z = randtn(n, X*beta, 1, a, b);
    %z = Xbeta+trandn(a-Xbeta, b-Xbeta);
        
    % draw beta and conditional on z
    
    B = inv(inv(B0) + X'*X);
    betabar = B * (B0\beta0 + X'*z);

    beta = mvnrnd(betabar, B)';
    
    if mod(m, 1000) == 0
        disp(['I have finished iteration', num2str(m), ' at ', datestr(now,'HH:MM:SS.FFF')]);
        disp(' beta is now: ');
        disp(beta);
    end;
    
    % storage
    betaM(m,:) = beta;
    
end

save results_probit;

%% plot and examine ESS
plot(betaM);
mean(betaM)

ESS1 = niters/(1+2*sum(autocorr(betaM(:,1), 100)));
ESS2 = niters/(1+2*sum(autocorr(betaM(:,2), 100)));
ESS3 = niters/(1+2*sum(autocorr(betaM(:,3), 100)));

%% compare with glm probit;
[b, dev, stats] = glmfit(X, y, 'binomial','link','probit', 'const', 'off');

