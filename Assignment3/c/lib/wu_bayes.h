/*******************************************************************************
 * Functions below are implementation of multivariate density and random number
 * functions using the GNU Scientific Library (GSL) functions.
 * Most of the functions are available in R. This program is the reimplementation
 * under GSL standard, mainly designed for the convience of use in pure C code.
 * The program is also a supplement for the existing functions in current GSL
 * library.
 *
 * There is a simlar program written by Ralph dos Santos Silva. I greatly 
 * acknowledge the help from reading his program.  This program is
 * more close to the taste and format of GSL library and have more
 * functions. 
 * 
 * Depends: GSL >= 1.12
 *
 * Copyright (C) 2010 Chunhua Wu
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307, USA.
 *
 *******************************************************************************/

#ifndef __WU_BAYES_H__
#define __WU_BAYES_H__
#include <gsl/gsl_rng.h>

#undef __BEGIN_DECLS
#undef __END_DECLS
#ifdef __cplusplus
# define __BEGIN_DECLS extern "C" {
# define __END_DECLS }
#else
# define __BEGIN_DECLS /* empty */
# define __END_DECLS /* empty */
#endif

__BEGIN_DECLS

int wu_linalg_kron (const gsl_matrix *A, const gsl_matrix *B, gsl_matrix *C);
int wu_bayes_panel(const gsl_rng *r, const int M0, const int M, const int G,
		   const int ml_method,
		   const gsl_vector *y, const gsl_matrix *X,
		   const gsl_matrix *W, const gsl_vector *ind,
		   const gsl_vector *beta0, const gsl_matrix *B0,
		   const double nu0, const double delta0,
		   const double rho0, const gsl_matrix *R0,
		   gsl_matrix *betam, gsl_matrix *bm, gsl_matrix *Dm, gsl_vector *sigma2m,
		   double *logmarglik);

int wu_bayes_panel_update(const gsl_rng *r, const gsl_vector *y, const gsl_matrix *X,
			  const gsl_matrix *W, const gsl_vector *ind,
			  const gsl_vector *beta0, const gsl_matrix *B0,
			  const double nu0, const double delta0, const double rho0,
			  const gsl_matrix *R0, int draw_beta, int draw_b, int draw_D, int draw_sigma2,
			  gsl_vector *beta, gsl_matrix *b, gsl_matrix *D,double *sigma2) ;

double wu_bayes_panel_loglik(const gsl_vector *y, const gsl_matrix *X, 
			     const gsl_matrix *W, const gsl_vector *ind,
			     const gsl_vector *beta, const gsl_matrix *D,
			     const double sigma2);

double wu_bayes_panel_logmarglik_chib95(const gsl_rng *r, const int M0, const int M, const int G,
					const gsl_vector *y, const gsl_matrix *X,
					const gsl_matrix *W, const gsl_vector *ind,
					const gsl_vector *beta0, const gsl_matrix *B0,
					const double nu0, const double delta0,
					const double rho0, const gsl_matrix *R0,
					const gsl_matrix *betam, const gsl_matrix *bm,
					const gsl_matrix *Dm, const gsl_vector *sigma2m);

double wu_bayes_panel_logmarglik_wu(const gsl_rng *r, const int M0, const int M,
				    const gsl_vector *y, const gsl_matrix *X,
				    const gsl_matrix *W, const gsl_vector *ind,
				    const gsl_vector *beta0, const gsl_matrix *B0,
				    const double nu0, const double delta0,
				    const double rho0, const gsl_matrix *R0,
				    const gsl_matrix *betam, const gsl_matrix *bm,
				    const gsl_matrix *Dm, const gsl_vector *sigma2m);


int wu_bayes_multireg_sample(const gsl_rng *r, const gsl_matrix *Y, const gsl_matrix *X,
			     const gsl_matrix *B0, const gsl_matrix *A0,
			     const double nu0, const gsl_matrix *V0, 
			     gsl_matrix *B, gsl_matrix *Sigma);
int wu_bayes_multireg_update_Sigma(const gsl_matrix *Y, const gsl_matrix *X,
				   const gsl_matrix *B0, const gsl_matrix *A0,
				   const double nu0, const gsl_matrix *V0, 
				   const gsl_matrix *B, double *nu,
				   gsl_matrix *V);
int wu_bayes_multireg_sample_Sigma(const gsl_rng *r,
				   const gsl_matrix *Y, const gsl_matrix *X,
				   const gsl_matrix *B0, const gsl_matrix *A0,
				   const double nu0, const gsl_matrix *V0, 
				   const gsl_matrix *B, gsl_matrix
				   *Sigma);
double wu_bayes_multireg_post_Sigma(const gsl_matrix *Sigmastar, 
				    const gsl_matrix *Y, const gsl_matrix *X,
				    const gsl_matrix *B0, const gsl_matrix *A0,
				    const gsl_matrix *B,
				    const double nu0, const gsl_matrix
				    *V0);
int wu_bayes_multireg_update_B(const gsl_matrix *Y, const gsl_matrix *X,
			       const gsl_matrix *B0, const gsl_matrix *A0,
			       gsl_matrix *Bhat, gsl_matrix *A);
int wu_bayes_multireg_sample_B(const gsl_rng *r,
			       const gsl_matrix *Y, const gsl_matrix *X,
			       const gsl_matrix *B0, const gsl_matrix *A0,
			       const gsl_matrix *Sigma, gsl_matrix
			       *B);
double wu_bayes_multireg_post_B(const gsl_matrix *Bstar,
				const gsl_matrix *Y, const gsl_matrix *X,
				const gsl_matrix *B0, const gsl_matrix *A0,
				const gsl_matrix *Sigma);


__END_DECLS

#endif /* __WU_BAYES_H__ */





