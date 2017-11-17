/********************************************************************************
 * BayesPanel estimation codes, rewritten on Feb 11, 2010 to improve efficiency.
 * Also included the marginal likelihood calculation using Chib 95 
 *
 *
 *
 ********************************************************************************/

#include<stdio.h>
#include<math.h>
#include<gsl/gsl_matrix.h>
#include<gsl/gsl_vector.h>
#include<gsl/gsl_randist.h>
#include<gsl/gsl_linalg.h>
#include<gsl/gsl_blas.h>
#include<gsl/gsl_statistics_double.h>
#include "wu_randist.h"
#include "wu_bayes.h"


/* codes for MCMC estimation for panel random effects model
   Allows for unbalanced panel,
   Yi = Xi*beta + Wi*bi + ei
   this is the implementation of alogrithm 2 of Chib and Carlin (1999)

 */
int wu_bayes_panel(const gsl_rng *r, const int M0, const int M, const int G,
		   const int ml_method,
		   const gsl_vector *y, const gsl_matrix *X,
		   const gsl_matrix *W, const gsl_vector *ind,
		   const gsl_vector *beta0, const gsl_matrix *B0,
		   const double nu0, const double delta0,
		   const double rho0, const gsl_matrix *R0,
		   gsl_matrix *betam, gsl_matrix *bm, gsl_matrix *Dm, gsl_vector *sigma2m,
		   double *logmarglik)
{
  const int n = ind->size;
  //const int nn = X->size1;
  const int k = X->size2;
  const int q = W->size2;

  gsl_vector *beta = gsl_vector_alloc(k);
  gsl_matrix *b = gsl_matrix_alloc(n, q);
  gsl_matrix *D = gsl_matrix_alloc(q, q);
  double *sigma2 = malloc(sizeof(double));
  gsl_vector_view v_D = gsl_vector_view_array(D->data, q*q);
  gsl_vector_view v_b = gsl_vector_view_array(b->data, n*q);


  int  iter;
  int draw_beta = 1;
  int draw_b = 1;
  int draw_D = 1;
  int draw_sigma2 = 1;


  //initialization:
  gsl_vector_memcpy(beta, beta0);
  wu_ran_invwishart(r, rho0, R0, D);
  *sigma2 = 1/gsl_ran_gamma(r, nu0/2.0, 2.0/delta0);
  
  for (iter=0; iter<M0+M; iter++)
    {
      wu_bayes_panel_update(r, y, X, W, ind, beta0, B0, nu0, delta0, rho0, R0, draw_beta, draw_b, draw_D, draw_sigma2, beta, b, D, sigma2);
      gsl_matrix_set_row(betam, iter, beta);
      gsl_matrix_set_row(bm, iter, &v_b.vector);
      gsl_matrix_set_row(Dm, iter, &v_D.vector);
      gsl_vector_set(sigma2m, iter, *sigma2);
    }

  if (ml_method == 1)
    {
      *logmarglik = wu_bayes_panel_logmarglik_chib95(r, M0, M, G, y, X, W, ind, beta0, B0, nu0, delta0,
						     rho0, R0, betam, bm, Dm, sigma2m);
    }
  
  if (ml_method == 2)
    {
      *logmarglik = wu_bayes_panel_logmarglik_wu(r, M0, M, y, X, W, ind, beta0, B0, nu0, delta0,
						     rho0, R0, betam, bm, Dm, sigma2m);
    }
  
  gsl_vector_free(beta);
  gsl_matrix_free(b);
  gsl_matrix_free(D);
  
  return 0;
}




//d_beta, d_b, d_D, d_sigma2 are ints to indicate wheather to draw corresponding values. 

int wu_bayes_panel_update(const gsl_rng *r, const gsl_vector *y, const gsl_matrix *X,
			  const gsl_matrix *W, const gsl_vector *ind,
			  const gsl_vector *beta0, const gsl_matrix *B0,
			  const double nu0, const double delta0, const double rho0,
			  const gsl_matrix *R0, int draw_beta, int draw_b, int draw_D, int draw_sigma2,
			  gsl_vector *beta, gsl_matrix *b, gsl_matrix *D,double *sigma2) 
{
  const int n = ind->size;
  const int nn = X->size1;
  const int k = X->size2;
  const int q = W->size2;

  int i, ni;
  int ind_row;
  double delta = 0;
 
  gsl_matrix *XVX = gsl_matrix_alloc(k, k);
  gsl_vector *XVy = gsl_vector_alloc(k);
  gsl_matrix_set_zero(XVX);
  gsl_vector_set_zero(XVy);

  gsl_matrix *B0inv = gsl_matrix_alloc(k, k);
  gsl_vector *betahat = gsl_vector_alloc(k);
  gsl_matrix *B = gsl_matrix_alloc(k, k);
  gsl_matrix_memcpy(B0inv, B0);
  gsl_linalg_cholesky_decomp(B0inv);
  gsl_linalg_cholesky_invert(B0inv); //create B0inv

  gsl_vector *bihat = gsl_vector_alloc(q);
  gsl_matrix *Dinv = gsl_matrix_alloc(q, q);
  gsl_matrix *Di = gsl_matrix_alloc(q, q);
  gsl_matrix *R = gsl_matrix_alloc(q, q);
  gsl_vector *eb = gsl_vector_alloc(nn); // eb = y - X\beta
  gsl_vector *e = gsl_vector_alloc(nn); // e = eb - Wb

  gsl_vector *work_k = gsl_vector_alloc(k);
  gsl_vector *work_q = gsl_vector_alloc(q);

  // calculate matrices XVX and XVy
  if (draw_beta)
    {
      ind_row = 0;
      for (i=0; i<n; i++)
	{
	  ni = gsl_vector_get(ind, i);
	  gsl_vector_const_view v_yi = gsl_vector_const_subvector(y, ind_row, ni);
	  gsl_matrix_const_view m_Xi = gsl_matrix_const_submatrix(X, ind_row, 0, ni, k);
	  gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
	  gsl_matrix *Vi = gsl_matrix_alloc(ni, ni);
	  gsl_matrix *work_ni_q = gsl_matrix_alloc(ni, q);
	  gsl_matrix *work_k_ni = gsl_matrix_alloc(k, ni);
	  gsl_matrix_set_identity(Vi);
	  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m_Wi.matrix,
			 D, 0.0, work_ni_q);
	  gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work_ni_q,
			 &m_Wi.matrix, *sigma2 , Vi);
	  gsl_linalg_cholesky_decomp(Vi);
	  gsl_linalg_cholesky_invert(Vi);

	  //completes Vi

	  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &m_Xi.matrix, Vi,
			 0.0, work_k_ni); //internal working matrix
	  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work_k_ni,
			 &m_Xi.matrix, 1.0, XVX);//completes XVX
	  gsl_blas_dgemv(CblasNoTrans, 1.0, work_k_ni, &v_yi.vector,
			 1.0, XVy); //completes XVy
	    
	  gsl_matrix_free(Vi);
	  gsl_matrix_free(work_ni_q);
	  gsl_matrix_free(work_k_ni);
	  ind_row += ni;
	}


      // 1.(a): update for beta|y, sigma2, D
      gsl_matrix_memcpy(B, B0inv);
      gsl_matrix_add(B, XVX);
      gsl_linalg_cholesky_decomp(B);
      gsl_linalg_cholesky_invert(B);//this compeltes B

      gsl_blas_dgemv(CblasNoTrans, 1.0, B0inv, beta0, 0.0, work_k);
      gsl_vector_add(work_k, XVy);
      gsl_blas_dgemv(CblasNoTrans, 1.0, B, work_k, 0.0, betahat);
      //this completes betahat

      wu_ran_mv_normal(r, betahat, B, beta);//update beta
    }

  if(draw_b)
    {
      gsl_vector_memcpy(eb, y);
      gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta, 1.0, eb);

      // 1.(b): update for bi
      ind_row = 0;
      for (i=0; i<n; i++)
	{
	  ni = gsl_vector_get(ind, i);
	  gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
	  gsl_vector_view v_bi = gsl_matrix_row(b, i);
	  gsl_matrix_memcpy(Di, D);
	  gsl_linalg_cholesky_decomp(Di);
	  gsl_linalg_cholesky_invert(Di);
	  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0/ *sigma2,
			 &m_Wi.matrix, &m_Wi.matrix, 1.0, Di);  // adjusted jan 2
	  gsl_linalg_cholesky_decomp(Di);
	  gsl_linalg_cholesky_invert(Di);

	  gsl_vector_view v_ebi = gsl_vector_subvector(eb, ind_row,  ni);
	  gsl_blas_dgemv(CblasTrans, 1.0, &m_Wi.matrix,
			 &v_ebi.vector, 0.0, work_q);
	  gsl_blas_dgemv(CblasNoTrans, 1.0/ *sigma2, Di, work_q, 0.0, bihat);
	  wu_ran_mv_normal(r, bihat, Di, &v_bi.vector);
	  ind_row += ni;
	}
    }

  if (draw_D)
    {
      //2. update for D;
      gsl_matrix_memcpy(R, R0);
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);
      for (i=0; i<n; i++)
	{
	  gsl_vector_view v_bi = gsl_matrix_row(b, i);
	  gsl_blas_dger(1.0, &v_bi.vector, &v_bi.vector, R);
	}
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);
      wu_ran_wishart(r, rho0+n, R, Dinv);
      gsl_matrix_memcpy(D, Dinv);
      gsl_linalg_cholesky_decomp(D);
      gsl_linalg_cholesky_invert(D);//D updated
    }

  if(draw_sigma2)
    {
      //3. update for sigma2;
      ind_row = 0;
      gsl_vector_memcpy(e, y);
      gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta, 1.0, e);
      for (i=0; i<n; i++)
	{
	  ni = gsl_vector_get(ind, i);
	  gsl_vector_view v_ei = gsl_vector_subvector(e, ind_row, ni);
	  gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
	  gsl_vector_view v_bi = gsl_matrix_row(b, i);
	  gsl_blas_dgemv(CblasNoTrans, -1.0, &m_Wi.matrix, &v_bi.vector, 1.0, &v_ei.vector);
	  ind_row += ni;
	}
  
      delta = delta0;
      for (i=0; i<nn; i++)
	{
	  delta += gsl_vector_get(e, i)*gsl_vector_get(e, i);
	}
      *sigma2 = 1/gsl_ran_gamma(r, (nu0+nn)/2.0, 2.0/delta);
    }

  gsl_matrix_free(XVX);
  gsl_vector_free(XVy);
  gsl_matrix_free(B0inv);
  gsl_vector_free(betahat);
  gsl_matrix_free(B);
  gsl_vector_free(bihat);
  gsl_matrix_free(Dinv);
  gsl_matrix_free(Di);
  gsl_matrix_free(R);
  gsl_vector_free(eb);
  gsl_vector_free(e);
  gsl_vector_free(work_k);
  gsl_vector_free(work_q);
  
  return 0;
}



double wu_bayes_panel_loglik(const gsl_vector *y, const gsl_matrix *X, 
			     const gsl_matrix *W, const gsl_vector *ind,
			     const gsl_vector *beta, const gsl_matrix *D,
			     const double sigma2)
{
  const int n = ind->size;
  const int nn = X->size1;
  //const int k = X->size2;
  const int q = W->size2;

  int i, ni;
  int ind_row = 0;

  gsl_vector *Xbeta = gsl_vector_alloc(nn);
  gsl_blas_dgemv(CblasNoTrans, 1.0, X, beta, 0.0, Xbeta);

  double loglik = 0.0;
  double pr;
  
  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_const_view v_yi = gsl_vector_const_subvector(y, ind_row, ni);
      gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_vector_view Xibeta = gsl_vector_subvector(Xbeta, ind_row, ni);
      gsl_matrix *Vi = gsl_matrix_alloc(ni, ni);
      gsl_matrix *work_ni_q = gsl_matrix_alloc(ni, q);

      gsl_matrix_set_identity(Vi);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m_Wi.matrix,
		     D, 0.0, work_ni_q);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work_ni_q,
		     &m_Wi.matrix, sigma2 , Vi);
      //completes Vi

      pr = wu_ran_mv_normal_pdf(&v_yi.vector, &Xibeta.vector, Vi);

      loglik += log(pr);

      gsl_matrix_free(Vi);
      gsl_matrix_free(work_ni_q);
      ind_row += ni;
    }

  gsl_vector_free(Xbeta);

  return loglik;
}

double wu_bayes_panel_logmarglik_chib95(const gsl_rng *r, const int M0, const int M, const int G,
					const gsl_vector *y, const gsl_matrix *X,
					const gsl_matrix *W, const gsl_vector *ind,
					const gsl_vector *beta0, const gsl_matrix *B0,
					const double nu0, const double delta0,
					const double rho0, const gsl_matrix *R0,
					const gsl_matrix *betam, const gsl_matrix *bm,
					const gsl_matrix *Dm, const gsl_vector *sigma2m)
{
  const int n = ind->size;
  const int nn = X->size1;
  const int k = X->size2;
  const int q = W->size2; 

  int i, j, iter;
  int ind_row = 0;
  int ni;
  
  gsl_vector *betastar = gsl_vector_alloc(k);
  gsl_matrix *Dstar = gsl_matrix_alloc(q, q);
  gsl_matrix *Dstarinv = gsl_matrix_alloc(q, q);
  gsl_matrix *R = gsl_matrix_alloc(q, q);
  
  gsl_matrix *b = gsl_matrix_alloc(n, q);
  gsl_vector *beta = gsl_vector_alloc(k);
  gsl_vector *betahat = gsl_vector_alloc(k);
  gsl_matrix *B = gsl_matrix_alloc(k, k);
  gsl_vector *e = gsl_vector_alloc(nn);
  double *sigma2 = malloc(sizeof(double)) ;

  gsl_matrix *B0inv = gsl_matrix_alloc(k, k);
  gsl_matrix *XVX = gsl_matrix_alloc(k, k);
  gsl_vector *XVy = gsl_vector_alloc(k);
  gsl_vector *work_k = gsl_vector_alloc(k);

  gsl_matrix_set_all(XVX, 0);
  gsl_vector_set_all(XVy, 0);
  gsl_matrix_memcpy(B0inv, B0);
  gsl_linalg_cholesky_decomp(B0inv);
  gsl_linalg_cholesky_invert(B0inv);
				     
  double sigma2star;

  double logmarglik = 0.0;
  double pD=0, psigma2=0, pbeta=0, pr=0;
  //calculate posterior mean, taken as the evaluation point
  for (i=0; i<k; i++)
    {
      betastar->data[i] = gsl_stats_mean(&betam->data[M0*betam->tda + i], betam->tda, M);
    }
  for (i=0; i<q*q; i++)
    {
      Dstar->data[i] = gsl_stats_mean(&Dm->data[M0*Dm->tda+i], Dm->tda, M);
    }
  sigma2star = gsl_stats_mean(&sigma2m->data[M0*sigma2m->stride], sigma2m->stride, M);
  gsl_matrix_memcpy(Dstarinv, Dstar);
  gsl_linalg_cholesky_decomp(Dstarinv);
  gsl_linalg_cholesky_invert(Dstarinv);

  logmarglik = wu_bayes_panel_loglik(y, X, W, ind, betastar, Dstar, sigma2star);
  pr = wu_ran_mv_normal_pdf(betastar, beta0, B0);
  logmarglik += log(pr);
  pr = wu_ran_wishart_pdf(Dstar, rho0, R0);
  logmarglik += log(pr);
  pr = gsl_ran_gamma_pdf(1/sigma2star, nu0/2, 2/delta0);
  logmarglik += log(pr);
  
  // calculate posterior from Gibbs output
  for (i=M0; i<M+M0; i++)
    {
      gsl_matrix_memcpy(R, R0);
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);
      for (j=0; j<n; j++)
	{
	  gsl_vector_const_view v_bi = gsl_vector_const_view_array(&bm->data[i*bm->tda+q*j], q);
	  gsl_blas_dger(1.0, &v_bi.vector, &v_bi.vector, R);
	}
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);
      pD += wu_ran_wishart_pdf(Dstarinv, rho0+n, R)/M;
    }

  int draw_beta = 1;
  int draw_b = 1;
  int draw_D = 0;
  int draw_sigma2 = 1;
  *sigma2 = sigma2star;
  // don not need to initialize beta and b, since beta|D,sigma2 
  for (iter=0; iter<G; iter++)
    {
      wu_bayes_panel_update(r, y, X, W, ind, beta0, B0, nu0, delta0, rho0, R0, draw_beta, draw_b
			    ,draw_D, draw_sigma2, beta, b, Dstar, sigma2);
      
      ind_row = 0;
      gsl_vector_memcpy(e, y);
      gsl_blas_dgemv(CblasNoTrans, -1.0, X, beta, 1.0, e);
      for (i=0; i<n; i++)
	{
	  ni = gsl_vector_get(ind, i);
	  gsl_vector_view v_ei = gsl_vector_subvector(e, ind_row, ni);
	  gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
	  gsl_vector_view v_bi = gsl_matrix_row(b, i);
	  gsl_blas_dgemv(CblasNoTrans, -1.0, &m_Wi.matrix, &v_bi.vector, 1.0, &v_ei.vector);
	  ind_row += ni;
	}
  
      double delta = delta0;
      for (i=0; i<nn; i++)
	{
	  delta += gsl_vector_get(e, i)*gsl_vector_get(e, i);
	}
      psigma2 += gsl_ran_gamma_pdf(1/sigma2star, (nu0+nn)/2, 2/delta)/G;
    }

  ind_row = 0;
  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_const_view v_yi = gsl_vector_const_subvector(y, ind_row, ni);
      gsl_matrix_const_view m_Xi = gsl_matrix_const_submatrix(X, ind_row, 0, ni, k);
      gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_matrix *Vi = gsl_matrix_alloc(ni, ni);
      gsl_matrix *work_ni_q = gsl_matrix_alloc(ni, q);
      gsl_matrix *work_k_ni = gsl_matrix_alloc(k, ni);
      gsl_matrix_set_identity(Vi);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m_Wi.matrix,
		     Dstar, 0.0, work_ni_q);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work_ni_q,
		     &m_Wi.matrix, sigma2star , Vi);
      gsl_linalg_cholesky_decomp(Vi);
      gsl_linalg_cholesky_invert(Vi);

      //completes Vi

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &m_Xi.matrix, Vi,
		     0.0, work_k_ni); //internal working matrix
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work_k_ni,
		     &m_Xi.matrix, 1.0, XVX);//completes XVX
      gsl_blas_dgemv(CblasNoTrans, 1.0, work_k_ni, &v_yi.vector,
		     1.0, XVy); //completes XVy
	    
      gsl_matrix_free(Vi);
      gsl_matrix_free(work_ni_q);
      gsl_matrix_free(work_k_ni);
      ind_row += ni;
    }

  // 1.(a): update for beta|y, sigma2, D
  gsl_matrix_memcpy(B, B0inv);
  gsl_matrix_add(B, XVX);
  gsl_linalg_cholesky_decomp(B);
  gsl_linalg_cholesky_invert(B);//this compeltes B

  gsl_blas_dgemv(CblasNoTrans, 1.0, B0inv, beta0, 0.0, work_k);
  gsl_vector_add(work_k, XVy);
  gsl_blas_dgemv(CblasNoTrans, 1.0, B, work_k, 0.0, betahat);
  //this completes betahat

  pbeta = wu_ran_mv_normal_pdf(betastar, betahat, B);//update beta

  logmarglik -= log(pD) + log(psigma2) + log(pbeta);

  gsl_vector_free(betastar);
  gsl_matrix_free(Dstar);
  gsl_matrix_free(Dstarinv);
  gsl_matrix_free(R);
  gsl_matrix_free(b);
  gsl_vector_free(beta);
  gsl_vector_free(betahat);
  gsl_matrix_free(B);
  free(sigma2);
  
  return logmarglik;
}
    

double wu_bayes_panel_logmarglik_wu(const gsl_rng *r, const int M0, const int M,
				    const gsl_vector *y, const gsl_matrix *X,
				    const gsl_matrix *W, const gsl_vector *ind,
				    const gsl_vector *beta0, const gsl_matrix *B0,
				    const double nu0, const double delta0,
				    const double rho0, const gsl_matrix *R0,
				    const gsl_matrix *betam, const gsl_matrix *bm,
				    const gsl_matrix *Dm, const gsl_vector *sigma2m)
{
  const int n = ind->size;
  const int nn = X->size1;
  const int k = X->size2;
  const int q = W->size2; 

  int i, j, iter;
  int ind_row = 0;
  int ni;
  
  gsl_vector *betastar = gsl_vector_alloc(k);
  gsl_matrix *Dstar = gsl_matrix_alloc(q, q);
  gsl_matrix *Dstarinv = gsl_matrix_alloc(q, q);
  gsl_matrix *R = gsl_matrix_alloc(q, q);
  
  gsl_matrix *b = gsl_matrix_alloc(n, q);
  gsl_vector *beta = gsl_vector_alloc(k);
  gsl_vector *betahat = gsl_vector_alloc(k);
  gsl_matrix *B = gsl_matrix_alloc(k, k);
  double sigma2star;

  gsl_vector *e = gsl_vector_alloc(nn);
  gsl_matrix *XVX = gsl_matrix_alloc(k, k);
  gsl_vector *XVy = gsl_vector_alloc(k);
  gsl_matrix *B0inv = gsl_matrix_alloc(k, k);
  gsl_vector *work_k = gsl_vector_alloc(k);

  gsl_matrix_set_all(XVX, 0);
  gsl_vector_set_all(XVy, 0);
  gsl_matrix_memcpy(B0inv, B0);
  gsl_linalg_cholesky_decomp(B0inv);
  gsl_linalg_cholesky_invert(B0inv);
  

  double logmarglik = 0.0;
  double pD=0, psigma2=0, pbeta=0, pr=0;
  //calculate posterior mean, taken as the evaluation point
  for (i=0; i<k; i++)
    {
      betastar->data[i] = gsl_stats_mean(&betam->data[M0*betam->tda + i], betam->tda, M);
    }
  for (i=0; i<q*q; i++)
    {
      Dstar->data[i] = gsl_stats_mean(&Dm->data[M0*Dm->tda+i], Dm->tda, M);
    }
  sigma2star = gsl_stats_mean(&sigma2m->data[M0*sigma2m->stride], sigma2m->stride, M);
  gsl_matrix_memcpy(Dstarinv, Dstar);
  gsl_linalg_cholesky_decomp(Dstarinv);
  gsl_linalg_cholesky_invert(Dstarinv);

  logmarglik = wu_bayes_panel_loglik(y, X, W, ind, betastar, Dstar, sigma2star);
  pr = wu_ran_mv_normal_pdf(betastar, beta0, B0);
  logmarglik += log(pr);
  pr = wu_ran_wishart_pdf(Dstar, rho0, R0);
  logmarglik += log(pr);
  pr = gsl_ran_gamma_pdf(1/sigma2star, nu0/2, 2/delta0);
  logmarglik += log(pr);
  
  // calculate posterior from Gibbs output
  for (i=M0; i<M0+M; i++)
    {
      gsl_matrix_memcpy(R, R0);
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);
      for (j=0; j<n; j++)
	{
	  gsl_vector_const_view v_bi = gsl_vector_const_view_array(&bm->data[i*bm->tda+q*j], q);
	  gsl_blas_dger(1.0, &v_bi.vector, &v_bi.vector, R);
	}
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);
      pD += wu_ran_wishart_pdf(Dstarinv, rho0+n, R)/M;
    }

  // don not need to initialize beta and b, since beta|D,sigma2

  double temp = 0.0;
  gsl_matrix *Rinv = gsl_matrix_alloc(q, q);
  gsl_matrix_memcpy(Rinv, Dstar);
  gsl_matrix_scale(Rinv, rho0+n-q-1);
  for (iter=M0; iter<M0+M; iter++)
    {
      gsl_matrix_const_view m_D = gsl_matrix_const_view_array(&Dm->data[iter*Dm->tda], q, q);
      gsl_vector_const_view v_beta = gsl_matrix_const_row(betam, iter);

      gsl_vector_memcpy(e, y);
      gsl_blas_dgemv(CblasNoTrans, -1.0, X, &v_beta.vector, 1.0, e);

      gsl_matrix_memcpy(R, R0);
      gsl_linalg_cholesky_decomp(R);
      gsl_linalg_cholesky_invert(R);

      ind_row = 0;
      for (i=0; i<n; i++)
	{
	  ni = gsl_vector_get(ind, i);
	  gsl_vector_view v_ei = gsl_vector_subvector(e, ind_row, ni);
	  gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
	  gsl_vector_const_view v_bi = gsl_vector_const_view_array(&bm->data[iter*bm->tda+q*i], q);
	  gsl_blas_dger(1.0, &v_bi.vector, &v_bi.vector, R);
	  gsl_blas_dgemv(CblasNoTrans, -1.0, &m_Wi.matrix, &v_bi.vector, 1.0, &v_ei.vector);
	  ind_row += ni;
	}
      //gsl_linalg_cholesky_decomp(R);
      //gsl_linalg_cholesky_invert(R);
      temp = wu_ran_invwishart_pdf(&m_D.matrix, rho0+n, R);
  
      double delta = delta0;
      for (i=0; i<nn; i++)
	{
	  delta += gsl_vector_get(e, i)*gsl_vector_get(e, i);
	}
      temp *= gsl_ran_gamma_pdf(1/sigma2star, (nu0+nn)/2, 2/delta);
      temp /= wu_ran_invwishart_pdf(&m_D.matrix, rho0+n, Rinv);
  
      psigma2 += temp/M;
    }

  //update pbeta
  ind_row = 0;
  for (i=0; i<n; i++)
    {
      ni = gsl_vector_get(ind, i);
      gsl_vector_const_view v_yi = gsl_vector_const_subvector(y, ind_row, ni);
      gsl_matrix_const_view m_Xi = gsl_matrix_const_submatrix(X, ind_row, 0, ni, k);
      gsl_matrix_const_view m_Wi = gsl_matrix_const_submatrix(W, ind_row, 0, ni, q);
      gsl_matrix *Vi = gsl_matrix_alloc(ni, ni);
      gsl_matrix *work_ni_q = gsl_matrix_alloc(ni, q);
      gsl_matrix *work_k_ni = gsl_matrix_alloc(k, ni);
      gsl_matrix_set_identity(Vi);
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, &m_Wi.matrix,
		     Dstar, 0.0, work_ni_q);
      gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1.0, work_ni_q,
		     &m_Wi.matrix, sigma2star , Vi);
      gsl_linalg_cholesky_decomp(Vi);
      gsl_linalg_cholesky_invert(Vi);

      //completes Vi

      gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, &m_Xi.matrix, Vi,
		     0.0, work_k_ni); //internal working matrix
      gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work_k_ni,
		     &m_Xi.matrix, 1.0, XVX);//completes XVX
      gsl_blas_dgemv(CblasNoTrans, 1.0, work_k_ni, &v_yi.vector,
		     1.0, XVy); //completes XVy
	    
      gsl_matrix_free(Vi);
      gsl_matrix_free(work_ni_q);
      gsl_matrix_free(work_k_ni);
      ind_row += ni;
    }

  // 1.(a): update for beta|y, sigma2, D
  gsl_matrix_memcpy(B, B0inv);
  gsl_matrix_add(B, XVX);
  gsl_linalg_cholesky_decomp(B);
  gsl_linalg_cholesky_invert(B);//this compeltes B

  gsl_blas_dgemv(CblasNoTrans, 1.0, B0inv, beta0, 0.0, work_k);
  gsl_vector_add(work_k, XVy);
  gsl_blas_dgemv(CblasNoTrans, 1.0, B, work_k, 0.0, betahat);
  //this completes betahat

  pbeta = wu_ran_mv_normal_pdf(betastar, betahat, B);//update beta

  logmarglik -= log(pD) + log(psigma2) + log(pbeta);

  gsl_vector_free(betastar);
  gsl_matrix_free(Dstar);
  gsl_matrix_free(Dstarinv);
  gsl_matrix_free(R);
  gsl_matrix_free(b);
  gsl_vector_free(beta);
  gsl_vector_free(betahat);
  gsl_matrix_free(B);
  gsl_matrix_free(Rinv);
  
  return logmarglik;
}
        














