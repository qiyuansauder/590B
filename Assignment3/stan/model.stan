//The stan model
functions {
  real obs_lpdf(row_vector process, vector theta){
    real out;
    real mu;
    real lambda;
    real b;
    real x;
    real tx;
    real T;

    mu = exp(theta[1]);
    lambda = exp(theta[2]);
    b = theta[3];
  
    x = process[1]-1;
    tx = process[2];
    T = process[3];
    out = x * log(lambda) - log(lambda+mu) + log(mu*exp(-(lambda+mu)*tx) + lambda *exp(-(lambda+mu)*T) + 1e-6);
    return out;
  }
}

data {
  int<lower=0> n; // number of users
  int<lower=0> nn; //number of observed transactions
  int<lower=0> k;
  int<lower=0> p;
  matrix[n, 3] process;
  //real x[n]; // number of transactions
  //real tx[n]; // later period transaction
  //real T[n]; //total number of observed periods
  real z[nn]; //transaction amount
  real t[nn]; //transaction time;
  int<lower=0> ind[nn]; //indicator for which customer;
  //matrix[n, k] XX; // observed demographics
  row_vector[k] XX[n];
  //vector[n] XX[n];
  cov_matrix[p] R; //initial variance matrix for Sigma;
}

parameters {
  row_vector[p] theta[n];
  real beta;
  real<lower=0> sigma2; 
  matrix[k, p] B;
  corr_matrix[p] Sigma;
  vector<lower=0>[p] tau;
}

// transformed parameters {
//   vector<lower=0>[p] tau;
//   for (i in 1:p)
//     tau[i] = 2.5 * tan(tau_unif[i]);
//   
//   
//   
//   
// }

model {
  vector[nn] bb;
  tau ~ cauchy(0, 2.5);
  sigma2 ~ cauchy(0,5);
  beta ~ normal(0, 10);
  to_vector(B) ~ normal(-2, 10);
  Sigma ~ lkj_corr(2);
  //Sigma ~ inv_wishart(4, R);
  
  {
    row_vector[p] u_theta[n];
    for (i in 1:n)
      u_theta[i] = XX[i] * B;
    theta ~ multi_normal(u_theta, quad_form_diag(Sigma, tau));
  }
  
  for (i in 1:n){
    //target += obs_lpdf(process[i]| theta[i]);
    process[i] ~ obs(theta[i]);
  }
    
  for (i in 1:nn){
    bb[i] = theta[ind[i]][3];
    log(z[i]) ~ normal(bb[i] + beta*log(t[i]), sigma2);
  }
}
