/*******************************************************************************
 * Full model estimation, using MCMC methods.
 * Use the panel model estimation procedure and multivariate regression procedure
 * Data structure used:
 *      record: x, tx, T;
 *      XX: time-invariant covariates 
 *      z: transaction amount
 *      X: time variant covariates
 *      W: random effect coefficients
 *      ind: indicator for number of transactions. = x+1
 * Version History
 *      Feb 28: first version
 *      Mar 23: added marginal likelihood calculation
 *      Oct 11: adjusted to allow one new variable
 *******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_statistics_double.h>
#include "lib/wu_bayes.h"
#include "lib/wu_randist.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

double fn_ln_marglik(const gsl_rng *r, const int M0, const int M,
		     const gsl_matrix *record, const gsl_matrix *XX,
		     const gsl_vector *z, const gsl_matrix *X,
		     const gsl_matrix *W, const gsl_vector *ind,
		     const gsl_matrix *G0, const gsl_matrix *A0,
		     const double mu0, const gsl_matrix *V0,
		     const gsl_vector *beta0, const gsl_matrix *B0,
		     const double nu0, const double delta0,
		     const gsl_matrix *Gm, const gsl_matrix *Sigmam,
		     const gsl_matrix *betam, const gsl_vector *simga2m,
		     const gsl_matrix *thetam);
int V_update(const double d, const double T,
	     const gsl_matrix *record, const gsl_vector *z, const gsl_vector *t,
	     const gsl_matrix *theta, const gsl_matrix *G, const gsl_matrix *Sigma,
	     const gsl_vector *beta, const double *sigma2,
	     gsl_vector *FV, gsl_vector *TV, gsl_vector *ETV);
int theta_sample(const gsl_rng *r, const gsl_matrix *record,
		 const gsl_matrix *XX, const gsl_vector *z,
		 const gsl_matrix *X, const gsl_matrix *W,
		 const gsl_vector *ind,
		 const gsl_matrix *G, const gsl_matrix *Sigma,
		 const gsl_vector *beta, const double *sigma2,
		 gsl_matrix *theta) ;
int eta_sample(const gsl_rng *r, const gsl_matrix *etabar,
	       const gsl_matrix *Sigma, const gsl_matrix *record,
	       gsl_matrix *eta);
double ln_lik_etai(const gsl_vector *etai, const gsl_vector *recordi);
double ln_lik_etai2(const gsl_vector *etai, const gsl_vector *recordi);
int b_sample(const gsl_rng *r, const gsl_matrix *bbar, const gsl_matrix *D,
	     const gsl_vector *y, const gsl_vector *ind,
	     const gsl_matrix *X, const gsl_matrix *W,
	     const gsl_vector *beta, const double sigma2,
	     gsl_matrix *b);
int beta_sigma2_sample(const gsl_rng *r, const gsl_vector *y, const gsl_vector *ind,
		       const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		       const gsl_vector *beta0, const gsl_matrix *B0,
		       const double nu0, const double delta0,
		       gsl_vector *beta, double *sigma2);
int sigma2_update(const gsl_vector *y, const gsl_vector *ind,
		  const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		  const double nu0, const double delta0,
		  const gsl_vector *beta, double *nu, double *delta);
int sigma2_sample(const gsl_rng *r, const gsl_vector *y, const gsl_vector *ind,
		  const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		  const double nu0, const double delta0,
		  const gsl_vector *beta, double *sigma2);
double sigma2_post(const double *sigma2star,
		   const gsl_vector *y, const gsl_vector *ind,
		   const gsl_matrix *X, const gsl_matrix *W,
		   const gsl_matrix *b,
		   const double nu0, const double delta0,
		   const gsl_vector *beta);
int beta_update(const gsl_vector *y, const gsl_vector *ind,
		const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		const gsl_vector *beta0, const gsl_matrix *B0,
		const double *sigma2, gsl_vector *betahat, gsl_matrix *B);
int beta_sample(const gsl_rng *r, const gsl_vector *y, const gsl_vector *ind,
		const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		const gsl_vector *beta0, const gsl_matrix *B0,
		const double *sigma2, gsl_vector *beta);
double beta_post(const gsl_vector *betastar,
		 const gsl_vector *y, const gsl_vector *ind,
		 const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		 const gsl_vector *beta0, const gsl_matrix *B0,
		 const double *sigma2);


int main (void)
{
    //define variables
    const int n = 408; // number of observations
    const int nn = 920;
    const int p = 5; // number of covariates in XX
    const int m = 2;
    const int k = 1; //number of covariates in X
    const int q = 1;// number of covariates in W
    const int M0 = 40000; // number of burn in values
    const int M = 10000; //numer of iterations
    
    int   iter;

    double d = 0.0043;//0.0043current discount 20% 0.0031//discount rate equals to yearly of 15%
    double T = 1000000; //GSL_POSINF
    
    const gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(r, 123456789);

    // read in data
    gsl_matrix *process = gsl_matrix_alloc(n , p+4); //first col is customer id
    gsl_matrix *amount = gsl_matrix_alloc(nn, k+4);
    //first col is customer id, second is transaction time,
    FILE *f = fopen("process.txt","r");
    gsl_matrix_fscanf(f, process);
    fclose(f);

    f = fopen("amount.txt","r");
    gsl_matrix_fscanf(f, amount);
    fclose(f);

    gsl_matrix_view record = gsl_matrix_submatrix(process, 0, 1, n, 3);
    gsl_matrix_view XX = gsl_matrix_submatrix(process, 0, 4, n, p);//time invariant variables
    gsl_vector_view t = gsl_matrix_column(amount, 1);
    gsl_vector_view z = gsl_matrix_column(amount, 2);
    gsl_matrix_view X = gsl_matrix_submatrix(amount, 0, 4, nn, k);//exclude intercept
    gsl_matrix_view W = gsl_matrix_submatrix(amount, 0, 3, nn, q);//intercept

    gsl_vector *ind = gsl_vector_alloc(n);
    gsl_matrix_get_col(ind, process, 1);
    gsl_vector_add_constant(ind, 1);
    
    //priors: p denotes process, to separate from amount
    gsl_matrix *G0 = gsl_matrix_alloc(p, m+q);
    gsl_matrix_set_all(G0, 0);
    gsl_matrix_set(G0, 0, 0, -4);
    gsl_matrix_set(G0, 0, 1, -4);
    gsl_matrix_set(G0, 0, 2, 4);
    gsl_matrix *A0 = gsl_matrix_alloc(p, p);
    gsl_matrix_set_identity(A0);
    gsl_matrix_scale(A0, 1);
    double mu0 = 4; 
    gsl_matrix *V0 = gsl_matrix_alloc(m+q, m+q);
    gsl_matrix_set_identity(V0);
    gsl_matrix_scale(V0, 2);

    gsl_vector *beta0 = gsl_vector_alloc(k);
    gsl_vector_set_all(beta0, 1);
    gsl_matrix *B0 = gsl_matrix_alloc(k, k);
    gsl_matrix_set_identity(B0);
    gsl_matrix_scale(B0, 2);
    double nu0 = 4;
    double delta0 = 2;

    //itermediate storage
    gsl_matrix *G = gsl_matrix_alloc(p, m+q);
    gsl_matrix *Sigma = gsl_matrix_alloc(m+q, m+q);
    gsl_vector *beta = gsl_vector_alloc(k);
    double *sigma2 = malloc(sizeof(double));
    gsl_vector_view v_G = gsl_vector_view_array(G->data, p*(m+q));
    gsl_vector_view v_Sigma = gsl_vector_view_array(Sigma->data, (m+q)*(m+q));

    //latent values
    gsl_matrix *theta = gsl_matrix_alloc(n, m+q);//latent values to augument
    gsl_vector_view v_theta = gsl_vector_view_array(theta->data, n*(m+q));
    gsl_matrix_view eta = gsl_matrix_submatrix(theta, 0, 0, n, m);
    gsl_matrix_view b = gsl_matrix_submatrix(theta, 0, m, n, q);

    gsl_vector *TV = gsl_vector_alloc(n);//expected customer life time value
    gsl_vector *FV = gsl_vector_alloc(n);
    gsl_vector *ETV = gsl_vector_alloc(pow(2, p-3)*6);
    
    //final output: big matrix
    gsl_matrix *Gm = gsl_matrix_alloc(M0+M, p*(m+q));
    gsl_matrix *betam = gsl_matrix_alloc(M0+M, k);
    gsl_vector *sigma2m = gsl_vector_alloc(M0+M);
    gsl_matrix *Sigmam = gsl_matrix_alloc(M0+M, (m+q)*(m+q));
    gsl_matrix *thetam = gsl_matrix_alloc(M, n*(m+q));
    gsl_matrix *TVm = gsl_matrix_alloc(M, n);
    gsl_matrix *FVm = gsl_matrix_alloc(M, n);
    gsl_matrix *ETVm = gsl_matrix_alloc(M, pow(2, p-3)*6); 

    //marginal likelihood values
    double ln_marglik = 0;
    
    // MCMC process
    // initialize values

    gsl_matrix_memcpy(G, G0);
    gsl_vector_memcpy(beta, beta0);
    *sigma2 = 1;
    gsl_matrix_set_identity(Sigma);
    gsl_matrix_set_all(&eta.matrix, -4);
    gsl_matrix_set_all(&b.matrix, 4);

    //estimation procedure
    for (iter=0; iter<M0+M; iter++)
      {
	printf("==========I am now in iteration %d==========\n", iter);
	//1. update theta
	theta_sample(r, &record.matrix, &XX.matrix, &z.vector,
		     &X.matrix, &W.matrix, ind, G, Sigma, beta,
		     sigma2, theta);
	
	//2. update G and Sigma
	wu_bayes_multireg_sample_Sigma(r, theta, &XX.matrix, G0, A0,
	mu0, V0, G, Sigma);
	wu_bayes_multireg_sample_B(r, theta, &XX.matrix, G0, A0,
				   Sigma, G);

	//3. update beta and sigma2
	beta_sigma2_sample(r, &z.vector, ind,&X.matrix, &W.matrix, &b.matrix,
			   beta0, B0, nu0, delta0, beta, sigma2);
		
	//calculate FV and TV if iter > M0
	if(iter>=M0) V_update(d, T, &record.matrix, &z.vector,
			      &t.vector, theta, G, Sigma, beta, sigma2, FV, TV, ETV);

	//output parameter estimates and EV
	gsl_matrix_set_row(Gm, iter, &v_G.vector);
	gsl_matrix_set_row(betam, iter, beta);
	gsl_vector_set(sigma2m, iter, *sigma2);
	gsl_matrix_set_row(Sigmam, iter, &v_Sigma.vector);
	if (iter >= M0)
	  {
	    gsl_matrix_set_row(thetam, iter-M0, &v_theta.vector);
	    gsl_matrix_set_row(TVm, iter-M0, TV);
	    gsl_matrix_set_row(FVm, iter-M0, FV);
	    gsl_matrix_set_row(ETVm, iter-M0, ETV);
	  }
      }

    //calculate marginal likelihood, reduced iterations
    ln_marglik = fn_ln_marglik(r, M0, M, &record.matrix, &XX.matrix, &z.vector,
			       &X.matrix, &W.matrix, ind, G0, A0, mu0,
			       V0, beta0, B0, nu0, delta0, Gm, Sigmam,
			       betam, sigma2m, thetam);
    printf("====ln marginal likelihood is %f====\n", ln_marglik);
        

    //write out estimation matrix
    f = fopen("est_G.txt", "wb");
    gsl_matrix_fprintf(f, Gm, "%lf");
    fclose(f);
    f = fopen("est_beta.txt", "wb");
    gsl_matrix_fprintf(f, betam, "%lf");
    fclose(f);
    f = fopen("est_sigma2.txt", "wb");
    gsl_vector_fprintf(f, sigma2m, "%lf");
    fclose(f);
    f = fopen("est_Sigma.txt", "wb");
    gsl_matrix_fprintf(f, Sigmam, "%lf");
    fclose(f);
    f = fopen("est_theta.txt", "wb");
    gsl_matrix_fprintf(f, thetam, "%lf");
    fclose(f);
    f = fopen("est_TV.txt", "wb");
    gsl_matrix_fprintf(f, TVm, "%lf");
    fclose(f);
    f = fopen("est_FV.txt", "wb");
    gsl_matrix_fprintf(f, FVm, "%lf");
    fclose(f);
    f = fopen("est_ETV.txt", "wb");
    gsl_matrix_fprintf(f, ETVm, "%lf");
    fclose(f);


    gsl_matrix_free(process);
    gsl_matrix_free(amount);
    gsl_vector_free(ind);
    gsl_matrix_free(G0);
    gsl_matrix_free(A0);
    gsl_matrix_free(V0);
    gsl_vector_free(beta0);
    gsl_matrix_free(B0);
    gsl_matrix_free(G);
    gsl_vector_free(beta);
    free(sigma2);
    gsl_matrix_free(Sigma);
    gsl_matrix_free(theta);
    gsl_vector_free(TV);
    gsl_vector_free(FV);
    gsl_matrix_free(Gm);
    gsl_matrix_free(betam);
    gsl_vector_free(sigma2m);
    gsl_matrix_free(Sigmam);
    gsl_matrix_free(thetam);
    gsl_matrix_free(TVm);
    gsl_matrix_free(FVm);
    gsl_matrix_free(ETVm);

    return 0;
}
    
    


//function for marginal likelihood calculation

double fn_ln_marglik(const gsl_rng *r, const int M0, const int M,
		     const gsl_matrix *record, const gsl_matrix *XX,
		     const gsl_vector *z, const gsl_matrix *X,
		     const gsl_matrix *W, const gsl_vector *ind,
		     const gsl_matrix *G0, const gsl_matrix *A0,
		     const double mu0, const gsl_matrix *V0,
		     const gsl_vector *beta0, const gsl_matrix *B0,
		     const double nu0, const double delta0,
		     const gsl_matrix *Gm, const gsl_matrix *Sigmam,
		     const gsl_matrix *betam, const gsl_vector *sigma2m,
		     const gsl_matrix *thetam)
{
  const int n = XX->size1;
  const int p = XX->size2;
  const int k = X->size2;
  const int m = 2;
  const int q = W->size2;
  int iter, i;

  gsl_matrix *Gstar = gsl_matrix_alloc(p, m+q);
  gsl_vector_view v_Gstar = gsl_vector_view_array(Gstar->data,
						  p*(m+q));
  gsl_vector_view v_G0 = gsl_vector_view_array(G0->data, p*(m+q));
  gsl_matrix *Sigmastar = gsl_matrix_alloc(m+q, m+q);
  gsl_vector_view v_Sigmastar = gsl_vector_view_array(Sigmastar->data, (m+q)*(m+q));
  gsl_vector *betastar = gsl_vector_alloc(k);
  double *sigma2star = malloc(sizeof(double));

  //intermediate storage for MCMC sampling
  gsl_matrix *theta = gsl_matrix_alloc(n, m+q);
  gsl_matrix_view b = gsl_matrix_submatrix(theta, 0, m, n, q);
  gsl_matrix *G = gsl_matrix_alloc(p, m+q);
  gsl_vector *beta = gsl_vector_alloc(k);
  double *sigma2 = malloc(sizeof(double));
      
  gsl_vector *thetai = gsl_vector_alloc(m+q);
  gsl_vector_view v_etai = gsl_vector_subvector(thetai, 0, m);
  gsl_vector_view v_bi = gsl_vector_subvector(thetai, m, q);
  gsl_matrix *thetastar = gsl_matrix_alloc(n, m+q);

  gsl_matrix *SA0 = gsl_matrix_alloc((m+q)*p, (m+q)*p);
				     
  double ln_marglik = 0.0;
  double lik_eta = 0.0, lik_b = 0.0;
  double ln_lik = 0.0, ln_pi_prior = 0.0;
  double post_G = 0.0, post_Sigma = 0.0, post_beta = 0.0, post_sigma2 = 0.0;

  double temp;
  int row = 0, ni;
  
  //calculate posterior mean
  for (i=0; i<p*(m+q); i++)
    {
      temp = gsl_stats_mean(&Gm->data[M0 * Gm->tda + i], Gm->tda, M);
      gsl_vector_set(&v_Gstar.vector, i, temp);
    }
  for (i=0; i<(m+q)*(m+q); i++)
    {
      temp = gsl_stats_mean(&Sigmam->data[M0*Sigmam->tda + i],
			    Sigmam->tda, M);
      gsl_vector_set(&v_Sigmastar.vector, i, temp);
    }
  for (i=0; i<k; i++)
    {
      temp = gsl_stats_mean(&betam->data[M0*betam->tda + i],
			    betam->tda, M);
      gsl_vector_set(betastar, i, temp);
    }
  *sigma2star = gsl_stats_mean(&sigma2m->data[M0*sigma2m->stride], sigma2m->stride,
			       M);

  FILE *fin = fopen("liklihood.txt", "wb");
  //calculate ln_lik, marginalized over thetai

  for (i=0; i<n; i++)
    {
      lik_eta = 0.0;
      lik_b = 0.0;
      ni = gsl_vector_get(ind, i);
      gsl_matrix_const_view Xi = gsl_matrix_const_submatrix(X, row, 0, ni, k);
      gsl_matrix_const_view Wi = gsl_matrix_const_submatrix(W, row, 0, ni, q);
      gsl_vector_const_view recordi = gsl_matrix_const_row(record, i);
      gsl_vector_const_view zi = gsl_vector_const_subvector(z, row, ni);
      gsl_vector *zistar = gsl_vector_alloc(ni);
      gsl_matrix *SS = gsl_matrix_alloc(ni, ni);
      gsl_matrix_set_identity(SS);
      gsl_matrix_scale(SS, *sigma2star);

      for (iter=0; iter<M; iter++)
	{
	  gsl_vector_const_view thetaiiter =
	    gsl_vector_const_view_array(&thetam->data[iter*thetam->tda+i*(m+q)],
					m+q);
	  gsl_vector_memcpy(thetai, &thetaiiter.vector);
	  
	  //lik_eta
	  lik_eta += exp(ln_lik_etai(&v_etai.vector, &recordi.vector))/M;
	  //lik_b
	  gsl_blas_dgemv(CblasNoTrans, 1.0, &Xi.matrix, betastar, 0.0,
			 zistar);
	  gsl_blas_dgemv(CblasNoTrans, 1.0, &Wi.matrix, &v_bi.vector,
			 1.0, zistar);
	  lik_b += wu_ran_mv_normal_pdf(&zi.vector,zistar, SS)/M;
	}
      ln_lik += log(lik_eta)+log(lik_b);
      gsl_vector_free(zistar);
      gsl_matrix_free(SS);
      row += ni;
      fprintf(fin, "====current ln_likelihood is %f %f %f====\n",
	      log(lik_eta), log(lik_b), ln_lik);
    }

  fclose(fin);
  
  ln_pi_prior += log(wu_ran_invwishart_pdf(Sigmastar, mu0, V0));
  wu_linalg_kron(Sigmastar, A0, SA0);
  ln_pi_prior += log(wu_ran_mv_normal_pdf(&v_Gstar.vector, &v_G0.vector,
				       SA0));
  printf("====log evaluated prior is %f====\n", ln_pi_prior);

  //post_Sigma

  for (iter=0; iter<M; iter++)
  {
      gsl_matrix_view m_theta =
	gsl_matrix_view_array(&thetam->data[iter*thetam->tda], n,
  m+q);
      gsl_matrix_view m_G =
	gsl_matrix_view_array(&Gm->data[iter*Gm->tda], p, m+q);
      
      post_Sigma += wu_bayes_multireg_post_Sigma(Sigmastar,
						 &m_theta.matrix, XX,
						 G0, A0, &m_G.matrix, mu0, V0)/M;
  }
  printf("====posterior for Sigma is %f====\n", post_Sigma);
  
  gsl_matrix_memcpy(G, Gstar);
  gsl_vector_memcpy(beta, betastar);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XX, Gstar, 0.0, theta);
  *sigma2 = *sigma2star;
  for (iter=0; iter<M; iter++)
  {
      theta_sample(r, record, XX, z, X, W, ind, G, Sigmastar, beta,
		   sigma2, theta);
      wu_bayes_multireg_sample_B(r, theta, XX, G0, A0, Sigmastar, G);
      beta_sigma2_sample(r, z, ind, X, W, &b.matrix, beta0, B0, nu0,
			 delta0, beta, sigma2);
      post_G += wu_bayes_multireg_post_B(Gstar, theta, XX, G0, A0, Sigmastar)/M;
  }
  printf("====posterior for G is %f====\n", post_G);

  gsl_vector_memcpy(beta, betastar);
  *sigma2 = *sigma2star;
  for (iter=0; iter<M; iter++)
  {
      theta_sample(r, record, XX, z, X, W, ind, Gstar, Sigmastar, beta,
		   sigma2, theta);
      beta_sigma2_sample(r, z, ind, X, W, &b.matrix, beta0, B0, nu0,
			 delta0, beta, sigma2);
      post_sigma2 += sigma2_post(sigma2star, z, ind, X, W, &b.matrix,
				 nu0, delta0, beta)/M; 
  }
  printf("====posterior for sigma2 is %f====\n", post_sigma2);

  gsl_vector_memcpy(beta, betastar);
  for (iter=0; iter<M; iter++)
  {
      theta_sample(r, record, XX, z, X, W, ind, Gstar, Sigmastar, betastar,
		   sigma2, theta);
      beta_sample(r, z, ind, X, W, &b.matrix, beta0, B0, sigma2star,beta);
      post_beta += beta_post(betastar, z, ind, X, W, &b.matrix, beta0,
			     B0, sigma2star)/M;
  }
  printf("====posterior for beta is %f====\n", post_beta);

  ln_marglik = ln_lik + ln_pi_prior - log(post_G) - log(post_Sigma) -
      log(post_sigma2) - log(post_beta);
  printf("====ln marginal likelihood is %f====\n", ln_marglik);

  FILE *f = fopen("marglik.txt","wb");
  fprintf(f,"====ln_likelihood is %f====\n", ln_lik);
  fprintf(f, "====log evaluated prior is %f====\n", ln_pi_prior);
  fprintf(f, "====posterior for Sigma is %f====\n", post_Sigma);
  fprintf(f, "====posterior for G is %f====\n", post_G);
  fprintf(f,"====posterior for sigma2 is %f====\n",  post_sigma2);
  fprintf(f, "====posterior for beta is %f====\n", post_beta);
  fprintf(f, "====ln marginal likelihood is %f====\n", ln_marglik);

  gsl_matrix_free(Gstar);
  gsl_matrix_free(Sigmastar);
  gsl_vector_free(betastar);
  free(sigma2star);
  gsl_matrix_free(theta);
  gsl_matrix_free(G);
  gsl_vector_free(beta);
  free(sigma2);
  gsl_vector_free(thetai);
  gsl_matrix_free(thetastar);
  gsl_matrix_free(SA0);
  
  return ln_marglik;
}




  
  




//function for Expected value update:
//FV is discounted future value, 
//TV is total value, discounted at time 0 (when acquired) value.


int V_update(const double d, const double T,
	     const gsl_matrix *record, const gsl_vector *z, const gsl_vector *t,
	     const gsl_matrix *theta, const gsl_matrix *G, const gsl_matrix *Sigma,
	     const gsl_vector *beta,
	     const double *sigma2,
	     gsl_vector *FV, gsl_vector *TV, gsl_vector *ETV)
{
  const int n = record->size1;
  //const int nn = z->size;
  const int p = G->size1;
  //const int k = beta->size;
  const int ncom = ETV->size;
  const int M = 1000;//numer of simulations in computing expectation

  // variables for ETV
  gsl_vector_const_view v_Gmu = gsl_matrix_const_column(G, 0);
  gsl_vector_const_view v_Glambda = gsl_matrix_const_column(G, 1);
  gsl_vector_const_view v_Gb = gsl_matrix_const_column(G, 2);
  double mu0, lambda0, b0, mu1, lambda1, b1, EV;
  gsl_vector *X = gsl_vector_alloc(p);
  gsl_vector *err = gsl_vector_alloc(3);
  double beta0, mv;
  int i, j, ind_row=0;
  double *temp = malloc(sizeof(double));
  gsl_matrix *TVm = gsl_matrix_alloc(M, ncom);
  gsl_vector *vec0 = gsl_vector_calloc(3);

  gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937);
  gsl_rng_set(r, 1234567);

  //variables for TV and FV
  double mui, lambdai, bi, tvi, fvi0, fvi;
  double zij, tij, ti0, tix, Ti;
  int ni;
  double pri;//probability of alive

  //calculate for ETV
  beta0 = gsl_vector_get(beta, 0);
  gsl_vector_set_all(X, 0);
  gsl_vector_set(X, 0, 1);
  for (i=0; i<M; i++)
    {
      wu_ran_mv_normal(r, vec0, Sigma, err);
      for (j=0; j<pow(2, p-3); j++)
	{
	  if (j==0)
	    {
	      gsl_vector_set(X, 2, 0);
	      gsl_vector_set(X, 3, 0);
	    }
	  if(j==1)
	    {
	      gsl_vector_set(X, 2, 0);
	      gsl_vector_set(X, 3, 1);
	    }
	  if(j==2)
	    {
	      gsl_vector_set(X, 2, 1);
	      gsl_vector_set(X, 3, 0);
	    }
	  if(j==3)
	    {
	      gsl_vector_set(X, 2, 1);
	      gsl_vector_set(X, 3, 1);
	    }
	  
	  //set values
	  gsl_vector_set(X, 1, 0);
	  gsl_blas_ddot(X, &v_Gmu.vector, temp);
	  mu0 = exp(*temp+gsl_vector_get(err,0));
	  gsl_blas_ddot(X, &v_Glambda.vector, temp);
	  lambda0 = exp(*temp+gsl_vector_get(err,1));
	  gsl_blas_ddot(X, &v_Gb.vector, temp);
	  b0 = *temp+gsl_vector_get(err,2);
      
	  gsl_vector_set(X, 1, 1);
	  gsl_blas_ddot(X, &v_Gmu.vector, temp);
	  mu1 = exp(*temp+gsl_vector_get(err,0));
	  gsl_blas_ddot(X, &v_Glambda.vector, temp);
	  lambda1 = exp(*temp+gsl_vector_get(err,1));
	  gsl_blas_ddot(X, &v_Gb.vector, temp);
	  b1= *temp+gsl_vector_get(err,2);
      
	  // all Non Google
	  EV = lambda0 * exp(b0 + *sigma2) * gsl_sf_gamma(beta0+1);
	  EV *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mu0+d)) / pow(mu0 + d, beta0+1);
	  gsl_matrix_set(TVm,i, j*6 + 0, EV);
	  // mu Google
	  EV = lambda0 * exp(b0 + *sigma2) * gsl_sf_gamma(beta0+1);
	  EV *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mu1+d)) / pow(mu1 + d, beta0+1);
	  gsl_matrix_set(TVm,i, j*6 + 1, EV);
	  // lambda Google
	  EV = lambda1 * exp(b0 + *sigma2) * gsl_sf_gamma(beta0+1);
	  EV *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mu0+d)) / pow(mu0 + d, beta0+1);
	  gsl_matrix_set(TVm,i, j*6 + 2, EV);
	  // mu and lambda Google
	  EV = lambda1 * exp(b0 + *sigma2) * gsl_sf_gamma(beta0+1);
	  EV *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mu1+d)) / pow(mu1 + d, beta0+1);
	  gsl_matrix_set(TVm,i, j*6 + 3, EV);
	  // b Google
	  EV = lambda0 * exp(b1 + *sigma2) * gsl_sf_gamma(beta0+1);
	  EV *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mu0+d)) / pow(mu0 + d, beta0+1);
	  gsl_matrix_set(TVm,i, j*6 + 4, EV);
	  // All Google 
	  EV = lambda1 * exp(b1 + *sigma2) * gsl_sf_gamma(beta0+1);
	  EV *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mu1+d)) / pow(mu1 + d, beta0+1);
	  gsl_matrix_set(TVm,i, j*6 + 5, EV);

	}
    }

  //averagig from TVm to ETV
  for (i=0; i<ncom; i++)
  {
    mv = gsl_stats_mean(&TVm->data[i],TVm->tda , M);
    gsl_vector_set(ETV, i, mv);
  }
        

  for (i=0; i<n; i++)
    {
      ni = (int) gsl_matrix_get(record, i, 0)+1;
      ti0 = gsl_vector_get (t, ind_row);
      tix = gsl_matrix_get(record, i, 1);
      Ti = gsl_matrix_get(record, i, 2);
      mui = exp(gsl_matrix_get(theta, i, 0));
      lambdai = exp(gsl_matrix_get(theta, i, 1));
      bi = gsl_matrix_get(theta, i, 2);
      beta0 = gsl_vector_get(beta,0);
      
      tvi = 0;
      for (j=0; j<ni; j++)
	{
	  zij = gsl_vector_get(z, ind_row+j);
	  tij = gsl_vector_get(t, ind_row+j);
	  tvi += exp(-(tij-ti0)*d) * exp(zij);
	}
      ind_row += ni;

      
      pri = (lambdai+mui) * (Ti-tix);
      pri = (mui/(lambdai+mui)) * (exp(pri)-1);
      pri = 1 / (1+pri);

      fvi = lambdai*exp(bi+ *sigma2/2);
      fvi *= gsl_sf_gamma(beta0+1)/pow(mui+d, beta0+1);
      fvi *= gsl_cdf_gamma_P(T, (beta0+1), 1/(mui+d)) -
      gsl_cdf_gamma_P(Ti, (beta0+1), 1/(mui+d));
      //may 5th,
      fvi *= exp((mui+d)*Ti);
      fvi *=pri;

      fvi0 = fvi*exp(-d*Ti);
      tvi += fvi0;

      gsl_vector_set(FV, i, fvi);
      gsl_vector_set(TV, i, tvi);
    }

  gsl_vector_free(X);
  gsl_matrix_free(TVm);
  gsl_vector_free(vec0);
  gsl_vector_free(err);
  free(temp);
  gsl_rng_free(r);
  
  return 0;
}
      

int theta_sample(const gsl_rng *r, const gsl_matrix *record,
		 const gsl_matrix *XX, const gsl_vector *z,
		 const gsl_matrix *X, const gsl_matrix *W,
		 const gsl_vector *ind,
		 const gsl_matrix *G, const gsl_matrix *Sigma,
		 const gsl_vector *beta, const double *sigma2,
		 gsl_matrix *theta) 
{
  const int n = XX->size1;
  const int m = 2;
  const int q = W->size2;
  int i;

  gsl_matrix_const_view S11 = gsl_matrix_const_submatrix(Sigma, 0, 0, m, m);
  gsl_matrix_const_view S12 = gsl_matrix_const_submatrix(Sigma, 0, m, m, q);
  gsl_matrix_const_view S21 = gsl_matrix_const_submatrix(Sigma, m, 0, q, m);
  gsl_matrix_const_view S22 = gsl_matrix_const_submatrix(Sigma, m, m, q, q);

  gsl_matrix *D11 = gsl_matrix_alloc(m, m);
  gsl_matrix *D22 = gsl_matrix_alloc(q, q);
  gsl_matrix *work_q_m = gsl_matrix_alloc(q, m);
  gsl_matrix *work_m_q = gsl_matrix_alloc(m, q);

  gsl_matrix *thetabar = gsl_matrix_alloc(n, m+q);
  gsl_matrix_view eta = gsl_matrix_submatrix(theta, 0, 0, n, m);
  gsl_matrix_view b = gsl_matrix_submatrix(theta, 0, m, n, q);
  gsl_matrix_view etabar = gsl_matrix_submatrix(thetabar, 0, 0, n, m);
  gsl_matrix_view bbar = gsl_matrix_submatrix(thetabar, 0, m, n, q);

  gsl_matrix *etahat = gsl_matrix_alloc(n, m);
  gsl_matrix *bhat = gsl_matrix_alloc(n, q);
  gsl_vector *work_m = gsl_vector_alloc(m);
  gsl_vector *work_q = gsl_vector_alloc(q);

  
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, XX, G, 0.0, thetabar);

  //update D11
  gsl_matrix_memcpy(D11, &S11.matrix);
  gsl_matrix_memcpy(D22, &S22.matrix);
  gsl_linalg_cholesky_decomp(D22);
  gsl_linalg_cholesky_invert(D22); // this is S22 ^-1
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &S12.matrix, D22, 0.0, work_m_q);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, work_m_q, &S21.matrix, 1.0, D11);
	
  //1. update eta|G, D11, etahat
  //a. update etahat = etabar + S12 S22^-1 (bi-bbari)
  for (i=0; i<n; i++)
    {
      gsl_vector_view etahati = gsl_matrix_row(etahat, i);
      gsl_vector_view etabari = gsl_matrix_row(&etabar.matrix, i);
      gsl_vector_view bi = gsl_matrix_row(&b.matrix, i);
      gsl_vector_view bbari = gsl_matrix_row(&bbar.matrix, i);
      gsl_vector_memcpy(work_q, &bi.vector);
      gsl_vector_sub(work_q, &bbari.vector);
      gsl_vector_memcpy(&etahati.vector, &etabari.vector);
      gsl_blas_dgemv(CblasNoTrans, 1.0, work_m_q, work_q, 1.0, &etahati.vector);
    }
  eta_sample(r, etahat, D11, record, &eta.matrix);

  //2. update b|G, D22, bhat
  gsl_matrix_memcpy(D22, &S22.matrix);
  gsl_matrix_memcpy(D11, &S11.matrix);
  gsl_linalg_cholesky_decomp(D11);
  gsl_linalg_cholesky_invert(D11);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &S21.matrix, D11, 0.0, work_q_m);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, work_q_m, &S12.matrix, 1.0, D22);

  //a. update bhat = bbar + S21 S11^-1(etai - etahat);
  for (i=0; i<n; i++)
    {
      gsl_vector_view bhati = gsl_matrix_row(bhat, i);
      gsl_vector_view bbari = gsl_matrix_row(&bbar.matrix, i);
      gsl_vector_view etai = gsl_matrix_row(&eta.matrix, i);
      gsl_vector_view etabari = gsl_matrix_row(&etabar.matrix, i);
      gsl_vector_memcpy(work_m, &etai.vector);
      gsl_vector_sub(work_m, &etabari.vector);
      gsl_vector_memcpy(&bhati.vector, &bbari.vector);
      gsl_blas_dgemv(CblasNoTrans, 1.0, work_q_m, work_m, 1.0, &bhati.vector);
    }
  b_sample(r, bhat, D22, z, ind, X, W, beta, *sigma2, &b.matrix);

  gsl_matrix_free(D11);
  gsl_matrix_free(D22);
  gsl_matrix_free(work_q_m);
  gsl_matrix_free(work_m_q);
  gsl_matrix_free(thetabar);
  gsl_matrix_free(etahat);
  gsl_matrix_free(bhat);
  gsl_vector_free(work_m);
  gsl_vector_free(work_q);

  return 0;
}



//independent Metropolis - Hastings updating for theta
//Use random walk metropolis

int eta_sample(const gsl_rng *r, const gsl_matrix *etabar,
	       const gsl_matrix *Sigma, const gsl_matrix *record,
	       gsl_matrix *eta)
{
  const int n = eta->size1;
  const int m = eta->size2;
  gsl_vector *etai_prop = gsl_vector_alloc(m);
  gsl_matrix *S_rw = gsl_matrix_alloc(m, m);
  gsl_matrix_set_identity(S_rw);
  gsl_matrix_scale(S_rw, 0.1);


  double alpha = 1.0;
  double u, temp;
  int accept = 0;
  int i;
  for (i=0; i<n; i++)
    {
      gsl_vector_const_view recordi = gsl_matrix_const_row(record, i);
      gsl_vector_const_view etaibar = gsl_matrix_const_row(etabar, i);
      gsl_vector_view etai = gsl_matrix_row(eta, i);
      wu_ran_mv_normal(r, &etaibar.vector, Sigma, etai_prop);

      temp = ln_lik_etai(etai_prop,  &recordi.vector)
	 -ln_lik_etai(&etai.vector, &recordi.vector);

      alpha = min(1, exp(temp));
      u = gsl_rng_uniform(r);
      if (u <= alpha)
	{
	  gsl_vector_memcpy(&etai.vector, etai_prop);
	  accept ++;
	}
       
    }
  printf("Accept is %d \n", accept);
  gsl_vector_free(etai_prop);
  gsl_matrix_free(S_rw);
  
  return 0;
}
  
  


//define the likelihood function, for tailoring issue.
double ln_lik_etai(const gsl_vector *etai, const gsl_vector *recordi)
{
    double lmu = gsl_vector_get(etai, 0);
    double llambda = gsl_vector_get(etai, 1);
    double mu = exp(lmu);
    double lambda = exp(llambda);
    double temp, ln_lik;

    double x = gsl_vector_get(recordi, 0);
    double tx = gsl_vector_get(recordi, 1);
    double T = gsl_vector_get(recordi, 2);

    //be careful with the log-liklihood function, when x = 0, lambda, mu small, can
    //esaily get to log_lik = 0
    //adjust x,


    temp = log(mu * exp(-(lambda+mu)*tx) + lambda *exp( -(lambda+mu)*T));

    //if (temp >= -2) temp =-1000;
    //printf("temp is %f\n", temp);
    //ln_lik = x*llambda - llambda - log(1 + exp(lmu - llambda)) +
    //temp;
    ln_lik = x*llambda - log(lambda+mu) + temp;
     //if (lambda>100)
    //lik = - 10000;

    if (lmu < -12 || llambda < -8 )
    ln_lik = -100;

    return ln_lik;

}

double ln_lik_etai2(const gsl_vector *etai, const gsl_vector *recordi)
{
    double lmu = gsl_vector_get(etai, 0);
    double llambda = gsl_vector_get(etai, 1);
    double mu = exp(lmu);
    double lambda = exp(llambda);
    double temp, ln_lik;

    double x = gsl_vector_get(recordi, 0);
    double tx = gsl_vector_get(recordi, 1);
    double T = gsl_vector_get(recordi, 2);

    //be careful with the log-liklihood function, when x = 0, lambda, mu small, can
    //esaily get to log_lik = 0
    //adjust x,


    temp = log(mu * exp(-(lambda+mu)*tx) + lambda *exp( -(lambda+mu)*T));

    //if (temp >= -2) temp =-1000;
    //printf("temp is %f\n", temp);
    //ln_lik = x*llambda - llambda - log(1 + exp(lmu - llambda)) +
    //temp;
    ln_lik = x*llambda - log(lambda+mu) + temp;
     //if (lambda>100)
    //lik = - 10000;

    return ln_lik;

}


//b update conditional on bhat, D22, z, X, beta, sigma2, 
int b_sample(const gsl_rng *r, const gsl_matrix *bbar, const gsl_matrix *D,
	     const gsl_vector *y, const gsl_vector *ind,
	     const gsl_matrix *X, const gsl_matrix *W,
	     const gsl_vector *beta, const double sigma2,
	     gsl_matrix *b)
{
  const int n = ind->size;
  const int nn = X->size1;
  const int q = W->size2;

  int i, ni, ind_row = 0;
  gsl_vector *bi = gsl_vector_alloc(q);
  gsl_vector *bhati = gsl_vector_alloc(q);
  gsl_matrix *Di = gsl_matrix_alloc(q, q);
  gsl_vector *work_q = gsl_vector_alloc(q);

  gsl_matrix *Dinv = gsl_matrix_alloc(q, q);
  gsl_matrix_memcpy(Dinv, D);
  gsl_linalg_cholesky_decomp(Dinv);
  gsl_linalg_cholesky_invert(Dinv);

  gsl_vector *z = gsl_vector_alloc(nn);
  gsl_vector_memcpy(z, y);
  gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta, 1.0, z);
  

  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_view zi = gsl_vector_subvector(z, ind_row, ni);
      gsl_matrix_const_view Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_vector_const_view bbari = gsl_matrix_const_row(bbar, i);

      gsl_matrix_memcpy(Di, Dinv);
      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0/sigma2, &Wi.matrix, &Wi.matrix,
		     1.0, Di);
      gsl_linalg_cholesky_decomp(Di);
      gsl_linalg_cholesky_invert(Di);//completes Di

      gsl_blas_dgemv(CblasNoTrans, 1.0, Dinv, &bbari.vector, 0.0, work_q);
      gsl_blas_dgemv(CblasTrans, 1.0/sigma2, &Wi.matrix, &zi.vector, 1.0, work_q);
      gsl_blas_dgemv(CblasNoTrans, 1.0, Di, work_q, 0.0, bhati);//completes bhati

      wu_ran_mv_normal(r, bhati, Di, bi);
      gsl_matrix_set_row(b, i, bi);
      ind_row += ni;
    }

  gsl_vector_free(bi);
  gsl_vector_free(bhati);
  gsl_matrix_free(Di);
  gsl_vector_free(work_q);
  gsl_matrix_free(Dinv);
  gsl_vector_free(z);

  return 0;
}

//beta update conditional on bi and zi, not marginalize over bi

int beta_sigma2_sample(const gsl_rng *r, const gsl_vector *y, const gsl_vector *ind,
		       const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		       const gsl_vector *beta0, const gsl_matrix *B0,
		       const double nu0, const double delta0,
		       gsl_vector *beta, double *sigma2)
{
  const int n = ind->size;
  const int nn = X->size1;
  const int k = X->size2;
  const int q = W->size2;

  gsl_vector *z = gsl_vector_alloc(nn);
  gsl_vector *e = gsl_vector_alloc(nn);
  gsl_matrix *B0inv = gsl_matrix_alloc(k, k);
  gsl_matrix *B = gsl_matrix_alloc(k, k);
  gsl_vector *betahat = gsl_vector_alloc(k);
  gsl_vector *work_k = gsl_vector_alloc(k);


  double *delta = malloc(sizeof(double));
  int i, ni, ind_row = 0;

  gsl_vector_memcpy(z, y);
  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_const_view bi = gsl_matrix_const_row(b, i);
      gsl_vector_view zi = gsl_vector_subvector(z, ind_row, ni);
      gsl_matrix_const_view Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_blas_dgemv(CblasNoTrans, -1.0, &Wi.matrix, &bi.vector, 1.0, &zi.vector);
      ind_row += ni;
    }//finish z update
  
  gsl_matrix_memcpy(B0inv, B0);
  gsl_linalg_cholesky_decomp(B0inv);
  gsl_linalg_cholesky_invert(B0inv);

  gsl_matrix_memcpy(B, B0inv);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1/ *sigma2, X, X, 1.0, B);
  gsl_linalg_cholesky_decomp(B);
  gsl_linalg_cholesky_invert(B);
  //completes B

  gsl_blas_dgemv(CblasNoTrans, 1.0, B0inv, beta0, 0.0, work_k);
  gsl_blas_dgemv(CblasTrans, 1.0/ *sigma2, X, z, 1.0, work_k);
  gsl_blas_dgemv(CblasNoTrans, 1.0, B, work_k, 0.0, betahat);

  wu_ran_mv_normal(r, betahat, B, beta);
  //complete beta update.

  gsl_vector_memcpy(e, z);
  gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta, 1.0, e);//adjusted Mar 28
  gsl_blas_ddot(e, e, delta);

  *sigma2 = 1 / gsl_ran_gamma(r, (nu0+nn)/2.0, 2.0/(delta0+ *delta));

  gsl_vector_free(z);
  gsl_vector_free(e);
  gsl_matrix_free(B0inv);
  gsl_matrix_free(B);
  gsl_vector_free(betahat);
  gsl_vector_free(work_k);
  free(delta);

  return 0;
}



int sigma2_update(const gsl_vector *y, const gsl_vector *ind,
		  const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		  const double nu0, const double delta0,
		  const gsl_vector *beta, double *nu, double *delta)
{
  const int n = ind->size;
  const int nn = X->size1;
  const int q = W->size2;

  gsl_vector *z = gsl_vector_alloc(nn);
  gsl_vector *e = gsl_vector_alloc(nn);

  int i, ni, ind_row = 0;

  gsl_vector_memcpy(z, y);
  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_const_view bi = gsl_matrix_const_row(b, i);
      gsl_vector_view zi = gsl_vector_subvector(z, ind_row, ni);
      gsl_matrix_const_view Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_blas_dgemv(CblasNoTrans, -1.0, &Wi.matrix, &bi.vector, 1.0, &zi.vector);
      ind_row += ni;
    }//finish z update
  
  gsl_vector_memcpy(e, z);
  gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta, 1.0, e);
  gsl_blas_ddot(e, e, delta);

  *nu = nu0 + nn;
  *delta += delta0;

  gsl_vector_free(z);
  gsl_vector_free(e);

  return 0;
}


int sigma2_sample(const gsl_rng *r, const gsl_vector *y, const gsl_vector *ind,
		  const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		  const double nu0, const double delta0,
		  const gsl_vector *beta, double *sigma2)
{
  double *delta = malloc(sizeof(double));
  double *nu = malloc(sizeof(double));
  
  sigma2_update(y, ind, X, W, b, nu0, delta0, beta, nu, delta);
  *sigma2 = 1 / gsl_ran_gamma(r, *nu/2.0, 2.0/ *delta);

  free(delta);
  free(nu);
  
  return 0;
}

double sigma2_post(const double *sigma2star,
		   const gsl_vector *y, const gsl_vector *ind,
		   const gsl_matrix *X, const gsl_matrix *W,
		   const gsl_matrix *b,
		   const double nu0, const double delta0,
		   const gsl_vector *beta)
{
  double post;
  double *delta = malloc(sizeof(double));
  double *nu = malloc(sizeof(double));

  sigma2_update(y, ind, X, W, b, nu0, delta0, beta, nu, delta);
  post = gsl_ran_gamma_pdf(1/ *sigma2star, *nu/2.0, 2.0/ *delta);

  free(delta);
  free(nu);
  
  return post;
}




int beta_update(const gsl_vector *y, const gsl_vector *ind,
		const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		const gsl_vector *beta0, const gsl_matrix *B0,
		const double *sigma2, gsl_vector *betahat, gsl_matrix *B)
{
  const int n = ind->size;
  const int nn = X->size1;
  const int k = X->size2;
  const int q = W->size2;

  gsl_vector *z = gsl_vector_alloc(nn);
  gsl_matrix *B0inv = gsl_matrix_alloc(k, k);
  gsl_vector *work_k = gsl_vector_alloc(k);

  int i, ni, ind_row = 0;

  gsl_vector_memcpy(z, y);
  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_const_view bi = gsl_matrix_const_row(b, i);
      gsl_vector_view zi = gsl_vector_subvector(z, ind_row, ni);
      gsl_matrix_const_view Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_blas_dgemv(CblasNoTrans, -1.0, &Wi.matrix, &bi.vector, 1.0, &zi.vector);
      ind_row += ni;
    }//finish z update
  
  gsl_matrix_memcpy(B0inv, B0);
  gsl_linalg_cholesky_decomp(B0inv);
  gsl_linalg_cholesky_invert(B0inv);

  gsl_matrix_memcpy(B, B0inv);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1/ *sigma2, X, X, 1.0, B);
  gsl_linalg_cholesky_decomp(B);
  gsl_linalg_cholesky_invert(B);
  //completes B update

  gsl_blas_dgemv(CblasNoTrans, 1.0, B0inv, beta0, 0.0, work_k);
  gsl_blas_dgemv(CblasTrans, 1.0/ *sigma2, X, z, 1.0, work_k);
  gsl_blas_dgemv(CblasNoTrans, 1.0, B, work_k, 0.0, betahat);

  //complete betahat update.

  gsl_vector_free(z);
  gsl_matrix_free(B0inv);
  gsl_vector_free(work_k);

  return 0;
}

int beta_sample(const gsl_rng *r, const gsl_vector *y, const gsl_vector *ind,
		const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		const gsl_vector *beta0, const gsl_matrix *B0,
		const double *sigma2, gsl_vector *beta)
{
  const int k = X->size2;
 

  gsl_matrix *B = gsl_matrix_alloc(k, k);
  gsl_vector *betahat = gsl_vector_alloc(k);

  beta_update(y, ind, X, W, b, beta0, B0, sigma2, betahat, B);

  wu_ran_mv_normal(r, betahat, B, beta);
  //complete beta update.

  gsl_matrix_free(B);
  gsl_vector_free(betahat);

  return 0;
}



double beta_post(const gsl_vector *betastar,
		 const gsl_vector *y, const gsl_vector *ind,
		 const gsl_matrix *X, const gsl_matrix *W, const gsl_matrix *b,
		 const gsl_vector *beta0, const gsl_matrix *B0,
		 const double *sigma2)
{
  const int k = X->size2;
  double post;
 
  gsl_matrix *B = gsl_matrix_alloc(k, k);
  gsl_vector *betahat = gsl_vector_alloc(k);

  beta_update(y, ind, X, W, b, beta0, B0, sigma2, betahat, B);
  post = wu_ran_mv_normal_pdf(betastar, betahat, B);
  //complete beta update.

  gsl_matrix_free(B);
  gsl_vector_free(betahat);

  return post;
}




  

  

  

  

  
  
  
  
  





/********************************************************************************
 * End of the estimation program, good luck!
 ********************************************************************************/













