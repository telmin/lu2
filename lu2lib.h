#include <timerlib.h>
void matmul_for_small_nk(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n);

void mydgemm(int m,
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

#ifndef USEGDR
void  gdrsetboardid(int boardid);
#endif
