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
e = random('logistic',0,1,[n 1]);
y = X*beta_true + e >0;

%% priors and storage
niters = 10000;
k = size(X,2);

beta0 = zeros(k, 1);
B0 = 10*eye(k);
beta = zeros(k, 1);
beta_prop = zeros(k,1);
V = 0.01*eye(k);
accept = 0; % acceptance rate;

betaM = zeros(niters, k);

%% doing the MCMC loop 
for m = 1:niters
    
    % random walk draw;
    beta_prop = mvnrnd(beta, V)';
    ll_prop = ll_logit(y, X, beta_prop);
    ll = ll_logit(y, X, beta);
    
    al = ll_prop - ll;
    al = al + log(mvnpdf(beta_prop, beta0, B0)) - log(mvnpdf(beta, beta0, B0));
    al = min(1.0, exp(al));
    
    draw = unifrnd(0,1);
    if (draw < al)
        beta = beta_prop;
        accept = accept + 1;
    end
    
    if mod(m, 100) == 0
        accept = accept/100;
        V = -1.19 * V / norminv(accept/2.0, 0.0, 1.0);
        disp(['Acceptance rate is ', num2str(accept)]);
        accept = 0;
    end
    
        
    if mod(m, 1000) == 0
        disp(['I have finished iteration', num2str(m), ' at ', datestr(now,'HH:MM:SS.FFF')]);
        disp(' beta is now: ');
        disp(beta);
    end;
    
    % storage
    betaM(m,:) = beta;
    
end

save results_logit;

%% plot and examine ESS
plot(betaM);
mean(betaM)

ESS1 = niters/(1+2*sum(autocorr(betaM(:,1), 100)));
ESS2 = niters/(1+2*sum(autocorr(betaM(:,2), 100)));
ESS3 = niters/(1+2*sum(autocorr(betaM(:,3), 100)));

%% compare with glm probit;
[b, dev, stats] = glmfit(X, y, 'binomial','link','logit', 'const', 'off');

