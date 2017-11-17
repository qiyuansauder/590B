/*******************************************************************************
 *The following program is designed to estimate the multivariate regression
 *model using MCMC sampling.
 *The model is
 *   Y = X B + E
 *   Y (n*m) X (n*k) B (k*m)
 *input:  B0 : k*m*1 vector
 *        A0:  k*m *k*m matrix
 *        nu0
 *        V0: m*m matrix
 *output: betam, (M+M0)* k*m
 *        Rm: (M+M0)*m*m
 *******************************************************************************/

#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include "wu_randist.h"
#include "wu_bayes.h"




//Model: Y = X\beta + e, 

int wu_bayes_multireg_sample(const gsl_rng *r, const gsl_matrix *Y, const gsl_matrix *X,
			     const gsl_matrix *B0, const gsl_matrix *A0,
			     const double nu0, const gsl_matrix *V0, 
			     gsl_matrix *B, gsl_matrix *Sigma)
{
    const int n = Y->size1;
    const int m = Y->size2;
    const int k = X->size2;
    double nu = nu0 + n;

    gsl_matrix *A0inv = gsl_matrix_alloc(k, k);
    gsl_matrix *A = gsl_matrix_alloc(k, k);
    gsl_matrix *V = gsl_matrix_alloc(m, m);
    gsl_matrix *Bhat = gsl_matrix_alloc(k, m);
    gsl_vector_view beta = gsl_vector_view_array(B->data, k*m);
    gsl_vector_view betahat = gsl_vector_view_array(Bhat->data, k*m);
    gsl_matrix *SA = gsl_matrix_alloc(k*m, k*m);
    gsl_matrix *work_k_m = gsl_matrix_alloc(k, m);
    gsl_matrix *work_m_k = gsl_matrix_alloc(m, k);
    gsl_matrix *work_n_m = gsl_matrix_alloc(n, m);

    gsl_matrix_memcpy(A0inv, A0);
    gsl_linalg_cholesky_decomp(A0inv);
    gsl_linalg_cholesky_invert(A0inv);

    gsl_matrix_memcpy(A, A0inv);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 1.0, A);
    gsl_linalg_cholesky_decomp(A);
    gsl_linalg_cholesky_invert(A);//completes A update

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, Y, 0.0, work_k_m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A0inv, B0, 1.0, work_k_m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, work_k_m, 0.0, Bhat);
    //completes Bhat update

    gsl_matrix_memcpy(work_n_m, Y);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, X, Bhat, 1.0, work_n_m);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, work_n_m, work_n_m, 0.0, V);
    gsl_matrix_memcpy(work_k_m, Bhat);
    gsl_matrix_sub(work_k_m, B0);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, work_k_m, A0inv, 0.0, work_m_k);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work_m_k, work_k_m, 1.0, V);
    gsl_matrix_add(V, V0);
    //completes V update

    //1. draw for Sigma
    wu_ran_invwishart(r, nu, V, Sigma);

    //2. draw for B
    wu_linalg_kron(Sigma, A, SA);
    wu_ran_mv_normal(r, &betahat.vector, SA, &beta.vector);

    gsl_matrix_free(A0inv);
    gsl_matrix_free(A);
    gsl_matrix_free(V);
    gsl_matrix_free(Bhat);
    gsl_matrix_free(SA);
    gsl_matrix_free(work_k_m);
    gsl_matrix_free(work_m_k);
    gsl_matrix_free(work_n_m);

    return 0;
}



int wu_bayes_multireg_update_Sigma(const gsl_matrix *Y, const gsl_matrix *X,
				   const gsl_matrix *B0, const gsl_matrix *A0,
				   const double nu0, const gsl_matrix *V0, 
				   const gsl_matrix *B, double *nu,
				   gsl_matrix *V)
{
    const int n = Y->size1;
    const int m = Y->size2;
    const int k = X->size2;
    *nu = nu0 + n;

    gsl_matrix *A0inv = gsl_matrix_alloc(k, k);
    gsl_matrix *A = gsl_matrix_alloc(k, k);
    gsl_matrix *Bhat = gsl_matrix_alloc(k, m);
    gsl_matrix *work_k_m = gsl_matrix_alloc(k, m);
    gsl_matrix *work_m_k = gsl_matrix_alloc(m, k);
    gsl_matrix *work_n_m = gsl_matrix_alloc(n, m);

    gsl_matrix_memcpy(A0inv, A0);
    gsl_linalg_cholesky_decomp(A0inv);
    gsl_linalg_cholesky_invert(A0inv);

    gsl_matrix_memcpy(A, A0inv);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 1.0, A);
    gsl_linalg_cholesky_decomp(A);
    gsl_linalg_cholesky_invert(A);//completes A update

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, Y, 0.0, work_k_m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A0inv, B0, 1.0, work_k_m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, work_k_m, 0.0, Bhat);
    //completes Bhat update

    gsl_matrix_memcpy(work_n_m, Y);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1.0, X, Bhat, 1.0, work_n_m);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, work_n_m, work_n_m, 0.0, V);
    gsl_matrix_memcpy(work_k_m, Bhat);
    gsl_matrix_sub(work_k_m, B0);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, work_k_m, A0inv, 0.0, work_m_k);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, work_m_k, work_k_m, 1.0, V);
    gsl_matrix_add(V, V0);
    //completes V update

    gsl_matrix_free(A0inv);
    gsl_matrix_free(A);
    gsl_matrix_free(Bhat);
    gsl_matrix_free(work_k_m);
    gsl_matrix_free(work_m_k);
    gsl_matrix_free(work_n_m);

    return 0;
}


int wu_bayes_multireg_sample_Sigma(const gsl_rng *r,
				   const gsl_matrix *Y, const gsl_matrix *X,
				   const gsl_matrix *B0, const gsl_matrix *A0,
				   const double nu0, const gsl_matrix *V0, 
				   const gsl_matrix *B, gsl_matrix *Sigma)
{
    const int m = Y->size2;
    double *nu = malloc(sizeof(double));

    gsl_matrix *V = gsl_matrix_alloc(m, m);
    
    //update  nu and V
    wu_bayes_multireg_update_Sigma(Y, X, B0, A0, nu0, V0, B, nu, V);

    //1. draw for Sigma
    wu_ran_invwishart(r, *nu, V, Sigma);

    free(nu);
    gsl_matrix_free(V);

    return 0;
}


double wu_bayes_multireg_post_Sigma(const gsl_matrix *Sigmastar, 
				    const gsl_matrix *Y, const gsl_matrix *X,
				    const gsl_matrix *B0, const gsl_matrix *A0,
				    const gsl_matrix *B,
				    const double nu0, const gsl_matrix *V0)
{
    const int m = Y->size2;
    double *nu = malloc(sizeof(double));
    double post;

    gsl_matrix *V = gsl_matrix_alloc(m, m);
    //update nu and V
    wu_bayes_multireg_update_Sigma(Y, X, B0, A0, nu0, V0, B, nu, V);
    post = wu_ran_invwishart_pdf(Sigmastar, *nu, V);

    free(nu);
    gsl_matrix_free(V);

    return post;
}



int wu_bayes_multireg_update_B(const gsl_matrix *Y, const gsl_matrix *X,
			       const gsl_matrix *B0, const gsl_matrix *A0,
			       gsl_matrix *Bhat, gsl_matrix *A)
{
    const int m = Y->size2;
    const int k = X->size2;

    gsl_matrix *A0inv = gsl_matrix_alloc(k, k);
    gsl_matrix *work_k_m = gsl_matrix_alloc(k, m);

    gsl_matrix_memcpy(A0inv, A0);
    gsl_linalg_cholesky_decomp(A0inv);
    gsl_linalg_cholesky_invert(A0inv);

    gsl_matrix_memcpy(A, A0inv);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, X, 1.0, A);
    gsl_linalg_cholesky_decomp(A);
    gsl_linalg_cholesky_invert(A);//completes A update

    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, X, Y, 0.0, work_k_m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A0inv, B0, 1.0, work_k_m);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, A, work_k_m, 0.0, Bhat);
    //completes Bhat update

    gsl_matrix_free(A0inv);
    gsl_matrix_free(work_k_m);

    return 0;
}


int wu_bayes_multireg_sample_B(const gsl_rng *r,
			       const gsl_matrix *Y, const gsl_matrix *X,
			       const gsl_matrix *B0, const gsl_matrix *A0,
			       const gsl_matrix *Sigma, gsl_matrix *B)
{
    const int m = Y->size2;
    const int k = X->size2;
    
    gsl_matrix *Bhat = gsl_matrix_alloc(k, m);
    gsl_matrix *A = gsl_matrix_alloc(k, k);
    gsl_vector_view v_B = gsl_vector_view_array(B->data, k*m);
    gsl_vector_view v_Bhat = gsl_vector_view_array(Bhat->data, k*m);
    gsl_matrix *SA = gsl_matrix_alloc(k*m, k*m);

    //update Bhat and A
    wu_bayes_multireg_update_B(Y, X, B0, A0, Bhat, A);

    //sample B
    wu_linalg_kron(Sigma, A, SA);
    wu_ran_mv_normal(r, &v_Bhat.vector, SA, &v_B.vector);

    gsl_matrix_free(A);
    gsl_matrix_free(Bhat);
    gsl_matrix_free(SA);

    return 0;
}


double wu_bayes_multireg_post_B(const gsl_matrix *Bstar,
				const gsl_matrix *Y, const gsl_matrix *X,
				const gsl_matrix *B0, const gsl_matrix *A0,
				const gsl_matrix *Sigma)
{
    const int m = Y->size2;
    const int k = X->size2;
    double post;
    
    gsl_matrix *Bhat = gsl_matrix_alloc(k, m);
    gsl_matrix *A = gsl_matrix_alloc(k, k);
    gsl_vector_view v_Bstar = gsl_vector_view_array(Bstar->data, k*m);
    gsl_vector_view v_Bhat = gsl_vector_view_array(Bhat->data, k*m);
    gsl_matrix *SA = gsl_matrix_alloc(k*m, k*m);

    //update Bhat and A
    wu_bayes_multireg_update_B(Y, X, B0, A0, Bhat, A);

    wu_linalg_kron(Sigma, A, SA);
    post = wu_ran_mv_normal_pdf(&v_Bstar.vector, &v_Bhat.vector, SA);

    gsl_matrix_free(A);
    gsl_matrix_free(Bhat);
    gsl_matrix_free(SA);

    return post;
}





//function to calculate kronecker product of two matrices

int wu_linalg_kron (const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C)
{
    int k, m, i, j;
    double v = 0;

    k = A->size1;
    m = B->size1;
    gsl_matrix *work = gsl_matrix_alloc(m, m);


    //C should be in dimension k*m
    for (i=0; i<k; i++)
    {
	for (j=0; j<k; j++){
	    gsl_matrix_view block = gsl_matrix_submatrix(C, i*m, j*m,
							 m, m);
	    v = gsl_matrix_get(A, i, j);
	    gsl_matrix_memcpy(work, B);
	    gsl_matrix_scale(work, v);
	    gsl_matrix_memcpy(&block.matrix, work);
	}
    }
    gsl_matrix_free(work);
    return 0;
}


      










      
  

  
   
  


























