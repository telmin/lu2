// lu2t.c
//
// test program for blocked LU decomposition
//
// Time-stamp: <09/08/31 00:41:24 makino>


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef NOBLAS
#include <cblas.h>
#endif
#include <lu2tlib.h>
void timer_init();
double cpusec();
double wsec();
#define FTYPE double
void copymats( int n, double a[n+1][n], double a2[n+1][n])
	      
{
    int i, j;
    for(j=0;j<n+1;j++)
	for(i=0;i<n;i++)
	    a2[j][i] = a[j][i];
}

void showresult(int n, double a[n+1][n], double x[])
{
    int i, j;
    double emax = 0;
    for(i=0;i<n;i++){
	int k;
	double b2=0;
	printf("%3d: ", i);
	//	for(j=0;j<n;j++) printf(" %10.3e", a[j][i]);
	for(j=0;j<n;j++) b2 += a[j][i] * x[j];
	double err = b2-a[n][i];
        emax =  (fabs(err) > emax) ? fabs(err):emax;
	printf(" %10.3e  %10.3e %10.3e %10.3e \n", x[i], a[n][i], b2, err);
    }
    printf("Emax= %10.3e\n", emax);
}
    
    
void readmat( int n, double a[n+1][n])
{
    int i, j;
    for(i=0;i<n;i++){
	for(j=0;j<n+1;j++) scanf("%le", &(a[j][i]));
    }
}

void printmat( int n, double a[n+1][n])

{
    int i, j;
    for(i=0;i<n;i++){
	printf("%3d: ", i);
	for(j=0;j<n+1;j++) printf(" %10.3e", a[j][i]);
	printf("\n");
    }
    printf("\n");
}

void backward_sub(int n,double a[n+1][n], double b[])
{
    int i,j,k;
    for (i=0;i<n;i++)b[i] = a[n][i];
    for(j=n-2;j>=0;j--)
	for(k=j+1;k<n;k++) b[j] -= b[k]*a[k][j];
}

    
void lu( int n, double a[n+1][n], double b[])
{
    int i, j, k;
    for(i=0;i<n-1;i++){
	// select pivot
	double amax = fabs(a[i][i]);
	int p=i;
	for(j=i+1;j<n;j++){
	    if (fabs(a[i][j]) > amax){
		amax = fabs(a[i][j]);
		p = j;
	    }
	}
	// exchange rows
	if (p != i){
	    for(j=i;j<n+1;j++){
		double tmp = a[j][p];
		a[j][p] = a[j][i];
		a[j][i]=tmp;
	    }
	}
		
	// normalize row i
	double ainv = 1.0/a[i][i];
	//	fprintf(stderr,"%d %e\n", i, ainv);
	for(k=i+1;k<n+1;k++) a[k][i]*= ainv;
	// subtract row i from all lower rows
	for(j=i+1;j<n;j++){
	    //	    fprintf(stderr,"j=%d \n",j);
	    for(k=i+1;k<n+1;k++) a[k][j] -= a[i][j] * a[k][i];
	}
    }
    printmat(n,a);
    
    a[n][n-1] /= a[n-1][n-1];
    backward_sub(n,a,b);
}


void lumcolumn( int n, double a[n+1][n], double b[], int m, int pv[],
		int recursive)
{
    int i;
    for(i=0;i<n;i+=m){
	if (recursive){
	    cm_column_decomposition_recursive(n, a+i, m, pv,i);
	}else{
	    cm_column_decomposition(n, a+i, m, pv,i);
	}
	cm_process_right_part(n,a+i,m,pv,i,n+1);
    }
    backward_sub(n,a,b);
}

main()
{
    int n;
    fprintf(stderr, "Enter n:");
    scanf("%d", &n);
    printf("N=%d\n", n);
#if 0    
    double a[n+1][n];
    double b[n];
    double acopy[n+1][n];
    double bcopy[n];
    int pv[n];
#endif
    double (*a)[];
    double (*acopy)[];
    int * pv;
    double *b, *bcopy;
    a = (double(*)[]) malloc(sizeof(double)*n*(n+1));
    acopy = (double(*)[]) malloc(sizeof(double)*n*(n+1));
    b = (double*)malloc(sizeof(double)*n);
    bcopy = (double*)malloc(sizeof(double)*n);
    pv = (int*)malloc(sizeof(int)*n);
    readmat(n,a);
    copymats(n,a,acopy);
    //    printmat(n,a,b);
    //    lu2columnv2(n,a,b);
    //    lu2columnv2(n,a,b);
    //    lub(n,a,b,NBK);
    //    printmat(n,a,b);
    //    showresult(n,acopy, b, bcopy);
    //    copymats(n,acopy,bcopy,a, b);
    //    lu(n,a,b);
    timer_init();
    lumcolumn(n,a,b,16,pv,1);
    double ctime=cpusec();
    double wtime=wsec();
    showresult(n,acopy, b);
    printf("cpsec =  %g wsec=%g\n",   ctime, wtime);
    return 0;
}
