#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <timerlib.h>
#ifndef NOBLAS
#ifdef MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif
#include "gdrdgemm.h"


#define NMAT 2048
void gdr_check_and_restart(double a[][NMAT],
			   double b[][NMAT], 
			   double c[][NMAT])
{
    int try =0;
    static int initialized = 0;
    if (initialized) return;
    while(1){
	int i,j;
	for(i=0;i<NMAT;i++){
	    for(j=0;j<NMAT;j++){
		a[i][j]=0;
		b[i][j]=i*NMAT+j;
		c[i][j]=0;
	    }
	}
	for(i=0;i<NMAT;i++)a[i][i]=1;
	cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
		     NMAT,NMAT, NMAT, 1.0, a, NMAT, b, NMAT, 0.0, c, NMAT);
	cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
		     NMAT,NMAT, NMAT, 1.0, a, NMAT, b, NMAT, 0.0, c, NMAT);
	mygdrdgemm(NMAT, NMAT, NMAT, 1.0, (double*)a, NMAT,
		   (double*)b, NMAT, 0.0, (double*) c, NMAT);
	int err = 0;
	for(i=0;i<NMAT;i++){
	    for(j=0;j<NMAT;j++){
		if (b[i][j] != c[i][j]){
		    err ++;
		}
	    }
	}
	if (err == 0){
	    fprintf(stderr,"gdr_check_and_restart passed %d\n", try);
	    initialized=1;
	    return;
	}
	try++;
	fprintf(stderr, "gdr_check_and_restart, err=%d try=%d\n", err, try);
	gdr_free();
	gdr_init();
    }
}
	
	   

	

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
)
{
    int NOTA, NOTB,gdrdoneflag;
    double alpha = ALPHA, beta = BETA;
    int F77M=M, F77N=N, F77K=K, F77lda=LDA, F77ldb=LDB, F77ldc=LDC;
    
    if(      TRANSA == CblasNoTrans ){
	NOTA = 1;
    }else{
	NOTA = 0;
    }
    if(      TRANSB == CblasNoTrans ){
	NOTB = 1;
    }else{
	NOTB = 0;
    }
    
    if( ORDER == CblasColMajor ){
	gdrdoneflag=0;
	gdr_dgemm_(&NOTA,&NOTB,&F77M,&F77N,&F77K, &alpha, &beta, 
		   &F77lda, &F77ldb, &F77ldc,A,B,C, &gdrdoneflag);
    }  else  {
	gdrdoneflag=0;
	gdr_dgemm_(&NOTA,&NOTB,&F77N,&F77M,&F77K, &alpha, &beta, 
		   &F77ldb, &F77lda, &F77ldc,B,A,C, &gdrdoneflag);
    }
    
    if(gdrdoneflag!=1){
	cblas_dgemm(ORDER, TRANSA, TRANSB,
		    M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);
    }
    /*
     * End of HPL_dgemm
     */
}
void dumpcmat(int m, int n, int nc, double c[][nc])
{
    static int callcount = 0;
    static FILE* fid;
    if (callcount == 0){
	fid = fopen("/tmp/matdata", "w");
    }
    callcount ++;
    if(callcount < 8){
	fprintf(fid,"\nPrint CMAT callcount=%d\n",callcount);
	int i, j;
	for(i=0;i<m;i++){
	    fprintf(fid,"\ni=%d\n", i);
	    for(j=0;j<n;j++){
		if ((j%8)==0 )fprintf(fid,"\n%5d:", j);
		fprintf(fid," %20.12e",c[i][j]);
	    }
	}
    }
}
double touchcmat(int m, int n, int nc, double c[][nc])
{
    double sum=0;
    int i, j;
    for(i=0;i<m;i++){
	for(j=0;j<n;j++){
	    sum += c[i][j]*c[i][j];
	}
    }
    return sum;
}

double ssum = 0.0;

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
		int nc)
{
    int nota=1, notb=1;
    int gdrdoneflag = 0;
    static int first_call = 1;
    if (first_call){
	gdrdgemm_set_procname(MP_myprocid());
	first_call=0;
	init_current_time();
    }
    //    char str[128];
    //    sprintf(str,"before sums = %25.20e %25.20e %25.20e",
    //	    touchcmat(m,k,na,a),  touchcmat(k,n,nb,b), touchcmat(m,n,nc,c));
    //    MP_message(str);
    //    dprintf(9,"mygdrdgemm omp_max_threads=%d procs=%d\n",
    //	    omp_get_max_threads(),omp_get_num_procs());
    double zero=0.0;
    //    int tmp=0;
    //    gdr_dgemm_(&nota, &notb, &n, &m, &k, &zero, &beta, &nb, &na, &nc,
    //	       b, a, c, &tmp);
    gdr_dgemm_(&nota, &notb, &n, &m, &k, &alpha, &beta, &nb, &na, &nc,
	       b, a, c, &gdrdoneflag);
    //	cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //		     m,n, k, alpha, a, na, b, nb, beta, c, nc);
    //	gdrdoneflag=1;
    //    fprintf(stderr,"gdrflag=%d\n", gdrdoneflag);
    if(gdrdoneflag!=1){
	cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
		     m,n, k, alpha, a, na, b, nb, beta, c, nc);
    }
    //    dumpcmat(m,n,nc,c);
    //    sprintf(str,"after sums = %25.20e %25.20e %25.20e",
    //	    touchcmat(m,k,na,a),  touchcmat(k,n,nb,b), touchcmat(m,n,nc,c));
    //    MP_message(str);
}
