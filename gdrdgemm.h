#ifndef _GDRDGEMM_H
#define _GDRDGEMM_H

#ifndef MKL
#include <cblas.h>
#else
#include <mkl_cblas.h>
#endif

#define NMAT 2048
void gdr_check_and_restart(double a[][NMAT],
			   double b[][NMAT], 
			   double c[][NMAT]);

void gdrblas_dgemm
(
#ifndef MKL 
   const enum CBLAS_ORDER             ORDER,
   const enum CBLAS_TRANSPOSE             TRANSA,
   const enum CBLAS_TRANSPOSE             TRANSB,
#else
   const CBLAS_ORDER             ORDER,
   const CBLAS_TRANSPOSE             TRANSA,
   const CBLAS_TRANSPOSE             TRANSB,
#endif   
   const int                        M,
   const int                        N,
   const int                        K,
   const double                     ALPHA,
   const double *                   A,
   const int                        LDA,
   const double *                   B,
   const int                        LDB,
   const double                     BETA,
   double *                         C,
   const int                        LDC
   );

void mygdrdgemm(int m,
		int n,
		int k,
		double alpha,
		double * a,
		int na,
		double * b,
		int nb,
		double beta,
		double * c,
		int nc);

#endif
