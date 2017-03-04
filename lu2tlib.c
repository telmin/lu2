// lu2tlub.c
//
//  blocked LU decomposition library for column-major version
//
// Time-stamp: <11/05/13 11:35:56 makino>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef NOBLAS
#ifdef MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif

#include <lu2lib.h>

typedef double v2df __attribute__((vector_size(16)));
typedef union {v2df v; double s[2];}v2u;

void timer_init();

#define MAXTHREADS 4

static int findpivot0(int n, double a[][n], int current)
{
    double amax = fabs(a[0][current]);
    int i;
    int p=current;
    BEGIN_TSC;
    for(i=current+1;i<n;i++){
	if (fabs(a[0][i]) > amax){
	    amax = fabs(a[0][i]);
	    p = i;
	}
    }
    END_TSC(t,7);
    return p;
}
static int findpivot(int n, double a[][n], int current)
{
    int p;
    int p2;
    BEGIN_TSC;
    p = cblas_idamax(n-current, a[0]+current, 1)+current;
    //    p2 = findpivot0( n, a, current);
    //    printf("n, current, p, p2 = %d %d %d %d\n",n, current,p, p2);
    END_TSC(t,7);
    return p;
}
static int findpivot_sequentical(int n, double a[][n], int current)
{
    double amax = fabs(a[0][current]);
    int i;
    int p=current;
    for(i=current+1;i<n;i++){
	if (fabs(a[0][i]) > amax){
	    amax = fabs(a[0][i]);
	    p = i;
	}
    }
    return p;
}
static int findpivot_omp(int n, double a[][n], int current)
// factor 2 slower than sequential code even for n=8k....
// on Core i7 920
{
    double amax[MAXTHREADS]={-1.0,-1.0,-1.0,-1.0};
    int p[MAXTHREADS];
    double am;
    int pm;
    int di= (n-current-1)/4;
    int k;
    BEGIN_TSC;
    if (di > 1024){
	//#pragma omp parallel for private(k)	
	for (k=0;k<MAXTHREADS; k++){
	    int i;
	    int istart = current+1+k*di;
	    int iend = current+1+(k+1)*di;
	    if (iend > n) iend = n;
	    for(i=istart;i<iend;i++){
		if (fabs(a[0][i]) > amax[k]){
		    amax[k] = fabs(a[0][i]);
		    p[k] = i;
		}
	    }
	}
	pm =p[0];
	am = amax[0];
	for (k=1;k<MAXTHREADS; k++){
	    if(amax[k]>am){
		am = amax[k];
		pm=p[k];
	    }
	}
    }else{
	pm=findpivot_sequentical( n,  a, current);
    }
    END_TSC(t,7);
    return pm;
}

static void swaprows(int n, double a[][n], int row1, int row2,
		     int cstart, int cend)
{
    int j;
    if (row1 != row2){
	for(j=cstart;j<cend;j++){
	    double tmp = a[j][row1];
	    a[j][row1] = a[j][row2];
	    a[j][row2]=tmp;
	}
    }
}
			   
static void scalerow( int n, double a[n+1][n], double scale,
	       int row, int cstart, int cend)
{
    int j;
    for(j=cstart;j<cend;j++) a[j][row]*= scale;
}


static void vsmulandsub(int n, double a[n+1][n], int cr, int cc,
		 int c0,  int r0,int r1)
{
    int j,k;
    double * ar = a[cr];
    k=c0;
    double s = a[c0][cc];
    double *al = a[c0];
    while (r0 & 7){
	al[r0] -= ar[r0]*s;
	r0++;
    }
    while (r1 & 7){
	al[r1-1] -= ar[r1-1]*s;
	r1--;
    }
    v2df * arv = (v2df*) (ar+r0);
    v2df * alv = (v2df*) (al+r0);
    v2df ss = (v2df){s,s};
    //    for(j=r0;j<r1;j++)
    //	al[j] -= ar[j]*s;
    for(j=0;j<(r1-r0)/2;j+=4){
	alv[j] -= arv[j]*ss;
	alv[j+1] -= arv[j+1]*ss;
	alv[j+2] -= arv[j+2]*ss;
	alv[j+3] -= arv[j+3]*ss;
	__builtin_prefetch(alv+j+32,1,3);
	__builtin_prefetch(arv+j+32,0,0);

    }
}

static void vvmulandsub(int n, double a[n+1][n], int cr, int cc,
		 int c0, int c1, int r0,int r1)
{
    int j,k;
    double * ar = a[cr];
#ifdef TIMETEST
    BEGIN_TSC;
#endif
    if (c1-c0 == 1){
	vsmulandsub(n,a, cr, cc,c0, r0,r1);
    }else{
	for (k=c0;k<c1;k++){
	    double s = a[k][cc];
	    double *al = a[k];
	    for(j=r0;j<r1;j++)
		al[j] -= ar[j]*s;
	}
    }
#ifdef TIMETEST
    END_TSC(t,6);
#endif    
}
static void mmmulandsub_old(int n, double a[n+1][n], int m0, int m1,
		 int c0, int c1, int r0,int r1)
{
    int j,k,l;
    printf("Enter mmul\n");
#ifndef NOBLAS
    cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
		 r1-r0, c1-c0, m1-m0, -1.0, &(a[m0][r0]), n,
		 &(a[c0][m0]), n, 1, &(a[c0][r0]), n );
    // example:
    // r0, m0 = i+m,i
    // m0, c0 = i, i+m
    // r0, c0 = i+m, i+m
    //r1-r0 = n-i-m
    // c1-c0 = iend-i-m
    // m1-m0 = m
#else
    for(j=r0;j<r1;j++)
	for (k=c0;k<c1;k++)
	    for (l=m0; l<m1; l++)
		a[k][j] -= a[l][j]*a[k][l];
#endif    
}


static void matmul_for_nk4_0(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int j,k,l;
    for(l=0;l<4;l+=2){
	for(k=0;k<4;k+=2){
	    for(j=0;j<n;j++){
		c[k][j] -= a[l][j]*b[k][l]+ a[l+1][j]*b[k][l+1];
		c[k+1][j] -= a[l][j]*b[k+1][l]+ a[l+1][j]*b[k+1][l+1];
	    }
	}
    }
}
static void matmul_for_nk8_0(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=8;
    double btmp[m][m];

    for(i=0;i<m;i++)for(j=0;j<m;j++)btmp[i][j]=b[i][j];
#pragma omp parallel for private(l,k,j)	
    for(l=0;l<8;l+=2){
	for(k=0;k<8;k+=2){
	    for(j=0;j<n;j++){
		c[k][j] -= a[l][j]*btmp[k][l]+ a[l+1][j]*btmp[k][l+1];
		c[k+1][j] -= a[l][j]*btmp[k+1][l]+ a[l+1][j]*btmp[k+1][l+1];
	    }
	}
    }

}
static void matmul_for_nk8_2(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=8;
    double btmp[m][m];

    for(i=0;i<m;i++)for(j=0;j<m;j++)btmp[i][j]=b[i][j];
#pragma omp parallel for private(l,k,j)	
    for(k=0;k<8;k+=2){
	for(l=0;l<8;l+=4){
	    for(j=0;j<n;j++){
		c[k][j] -= a[l][j]*btmp[k][l]
		    + a[l+1][j]*btmp[k][l+1]
		    + a[l+2][j]*btmp[k][l+2]
		    + a[l+3][j]*btmp[k][l+3];
		c[k+1][j] -= a[l][j]*btmp[k+1][l]
		    + a[l+1][j]*btmp[k+1][l+1]
		    + a[l+2][j]*btmp[k+1][l+2]
		    + a[l+3][j]*btmp[k+1][l+3];
		//		c[k][j] -= a[l][j]*btmp[k][l]+ a[l+1][j]*btmp[k][l+1];
		//		c[k+1][j] -= a[l][j]*btmp[k+1][l]+ a[l+1][j]*btmp[k+1][l+1];
	    }
	}
    }

}
static void matmul_for_nk8_3(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=8;
    const int mm = 32;
    double btmp[m][m];
    double atmp[m][mm];

    for(i=0;i<m;i++)for(j=0;j<m;j++)btmp[i][j]=b[i][j];
    for(j=0;j<n;j+=mm){
	int jj;
	for(i=0;i<m;i++){
	    for(k=0;k<mm;k++){
		atmp[i][k]=a[i][j+k];
	    }
	}
	for(i=0;i<m;i++){
	    int jjend = mm;
	    if (jjend+j > n) jjend = n-j;
	    for(jj=0;jj<jjend;jj+=2){
		c[i][j+jj] -= atmp[0][jj]*btmp[i][0]
		    +atmp[1][jj]*btmp[i][1]
		    +atmp[2][jj]*btmp[i][2]
		    +atmp[3][jj]*btmp[i][3]
		    +atmp[4][jj]*btmp[i][4]
		    +atmp[5][jj]*btmp[i][5]
		    +atmp[6][jj]*btmp[i][6]
		    +atmp[7][jj]*btmp[i][7];
		c[i][j+jj+1] -= atmp[0][jj+1]*btmp[i][0]
		    +atmp[1][jj+1]*btmp[i][1]
		    +atmp[2][jj+1]*btmp[i][2]
		    +atmp[3][jj+1]*btmp[i][3]
		    +atmp[4][jj+1]*btmp[i][4]
		    +atmp[5][jj+1]*btmp[i][5]
		    +atmp[6][jj+1]*btmp[i][6]
		    +atmp[7][jj+1]*btmp[i][7];
	    }
	}
    }

}
static void matmul_for_nk8_4(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=8;
    const int mm = 16;
    double btmp[m][m];
    v2df b2tmp[m][m];
    double atmp[m][mm];

    for(i=0;i<m;i++)for(j=0;j<m;j++)b2tmp[i][j]=(v2df){b[i][j],b[i][j]};
    //#pragma omp parallel for private(j,i,atmp)
    //use of OMP here does not speed things up
    for(j=0;j<n;j+=mm){
	int jj;
	for(i=0;i<m;i++){
	    v2df * dest = (v2df*)(atmp[i]);
	    v2df * src = (v2df*)(a[i]+j);
	    dest[0]=src[0];
	    dest[ 1]=src[ 1];
	    dest[ 2]=src[ 2];
	    dest[ 3]=src[ 3];
	    dest[ 4]=src[ 4];
	    dest[ 5]=src[ 5];
	    dest[ 6]=src[ 6];
	    dest[ 7]=src[ 7];
	}
	for(i=0;i<m;i++){
	    int jjend = mm;
	    __builtin_prefetch(c[i]+j+64,1,0);
	    __builtin_prefetch(c[i]+j+80,1,0);
	    if (jjend+j > n) jjend = n-j;
	    for(jj=0;jj<jjend;jj+=4){
		v2df* cp = (v2df*)(&c[i][j+jj]);
		v2df* ap = (v2df*)(atmp[0]+jj);
		v2df* cpp = (v2df*)(&c[i][j+jj+2]);
		v2df* app = (v2df*)(atmp[0]+jj+2);
		*cp -= (*ap)*b2tmp[i][0]
		    +(*(ap+16))*b2tmp[i][1]
		    +(*(ap+32))*b2tmp[i][2]
		    +(*(ap+48))*b2tmp[i][3]
		    +(*(ap+64))*b2tmp[i][4]
		    +(*(ap+80))*b2tmp[i][5]
		    +(*(ap+96))*b2tmp[i][6]
		    +(*(ap+112))*b2tmp[i][7];
		*cpp -= (*app)*b2tmp[i][0]
		    +(*(app+16))*b2tmp[i][1]
		    +(*(app+32))*b2tmp[i][2]
		    +(*(app+48))*b2tmp[i][3]
		    +(*(app+64))*b2tmp[i][4]
		    +(*(app+80))*b2tmp[i][5]
		    +(*(app+96))*b2tmp[i][6]
		    +(*(app+112))*b2tmp[i][7];
	    }
	}
    }

}
static void matmul_for_nk8_5(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=8;
    const int mm = 32;
    double btmp[m][m];
    v2df b2tmp[m][m];

    for(i=0;i<m;i++)for(j=0;j<m;j++)b2tmp[i][j]=(v2df){b[i][j],b[i][j]};
    //#pragma omp parallel for private(j,i)
    //use of OMP here does not speed things up
    for(j=0;j<n;j+=mm){
	double atmp[m][mm];
	int jj;
	for(i=0;i<m;i++){
	    v2df * dest = (v2df*)(atmp[i]);
	    v2df * src = (v2df*)(a[i]+j);
	    dest[0]=src[0];
	    dest[ 1]=src[ 1];
	    dest[ 2]=src[ 2];
	    dest[ 3]=src[ 3];
	    dest[ 4]=src[ 4];
	    dest[ 5]=src[ 5];
	    dest[ 6]=src[ 6];
	    dest[ 7]=src[ 7];
	    dest[ 8]=src[ 8];
	    dest[ 9]=src[ 9];
	    dest[ 10]=src[ 10];
	    dest[ 11]=src[ 11];
	    dest[ 12]=src[ 12];
	    dest[ 13]=src[ 13];
	    dest[ 14]=src[ 14];
	    dest[ 15]=src[ 15];
	}
	for(i=0;i<m;i++){
	    int jjend = mm;
	    __builtin_prefetch(c[i]+j+64,1,0);
	    __builtin_prefetch(c[i]+j+80,1,0);
	    if (jjend+j > n) jjend = n-j;
	    for(jj=0;jj<jjend;jj+=4){
		v2df* cp = (v2df*)(&c[i][j+jj]);
		v2df* ap = (v2df*)(atmp[0]+jj);
		v2df* cpp = (v2df*)(&c[i][j+jj+2]);
		v2df* app = (v2df*)(atmp[0]+jj+2);
		*cp -= (*ap)*b2tmp[i][0]
		    +(*(ap+16))*b2tmp[i][1]
		    +(*(ap+32))*b2tmp[i][2]
		    +(*(ap+48))*b2tmp[i][3]
		    +(*(ap+64))*b2tmp[i][4]
		    +(*(ap+80))*b2tmp[i][5]
		    +(*(ap+96))*b2tmp[i][6]
		    +(*(ap+112))*b2tmp[i][7];
		*cpp -= (*app)*b2tmp[i][0]
		    +(*(app+16))*b2tmp[i][1]
		    +(*(app+32))*b2tmp[i][2]
		    +(*(app+48))*b2tmp[i][3]
		    +(*(app+64))*b2tmp[i][4]
		    +(*(app+80))*b2tmp[i][5]
		    +(*(app+96))*b2tmp[i][6]
		    +(*(app+112))*b2tmp[i][7];
	    }
	}
    }

}

static void matmul_for_nk8(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int m0,m1,m2,m3;
    int s1, s2, s3;
    int ds = (n/64)*16;
    s1 = ds;
    s2 = ds*2;
    s3 = ds*3;
    m0 = ds;
    m1 = ds;
    m2 = ds;
    m3 = n-s3;
    //    fprintf(stderr,"n, s, m = %d %d %d %d %d %d %d %d\n",
    //    n,s1,s2,s3,m0,m1,m2,m3);
    
#pragma omp parallel
#pragma omp sections
  {
#pragma omp section
      matmul_for_nk8_5(n1,a,n2,b,n3,c,m0);
#pragma omp section
      matmul_for_nk8_5(n1,((double*)a)+s1 ,n2,b,n3,((double*)c)+s1,m1);
#pragma omp section
      matmul_for_nk8_5(n1,((double*)a)+s2 ,n2,b,n3,((double*)c)+s2,m2);
#pragma omp section
      matmul_for_nk8_5(n1,((double*)a)+s3 ,n2,b,n3,((double*)c)+s3,m3);
  }
}

static void matmul_for_nk4_3(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=4;
    const int mm = 16;
    double btmp[m][m];
    v2df b2tmp[m][m];
    double atmp[m][mm];
    v2df atmpt[mm/2][m];

    for(i=0;i<m;i++)for(j=0;j<m;j++)b2tmp[i][j]=(v2df){b[i][j],b[i][j]};
    //#pragma omp parallel for private(j,i,atmp)
    //use of OMP here does not speed things up
    for(j=0;j<n;j+=mm){
	int jj;

	for(i=0;i<m;i++){
	    v2df * dest = (v2df*)(atmp[i]);
	    v2df * src = (v2df*)(a[i]+j);
	    atmpt[ 0][i]=src[0];
	    atmpt[ 1][i]=src[ 1];
	    atmpt[ 2][i]=src[ 2];
	    atmpt[ 3][i]=src[ 3];
	    atmpt[ 4][i]=src[ 4];
	    atmpt[ 5][i]=src[ 5];
	    atmpt[ 6][i]=src[ 6];
	    atmpt[ 7][i]=src[ 7];
	}
	for(i=0;i<m;i++){
	    int jjend = mm;
	    if (jjend+j > n) jjend = n-j;
	    v2df* cp = (v2df*)(&c[i][j]);
	    v2df* ap = (v2df*)(atmpt[0]);
	    v2df* cpp = (v2df*)(&c[i][j+2]);
	    v2df* app = (v2df*)(atmpt[1]);
	    v2df* bp = b2tmp[i];
	    for(jj=0;jj<jjend;jj+=4){
		*cp -= ap[0]*bp[0]
		    +ap[1]*bp[1]
		    +ap[2]*bp[2]
		    +ap[3]*bp[3];
		*cpp -= app[0]*bp[0]
		    +app[1]*bp[1]
		    +app[2]*bp[2]
		    +app[3]*bp[3];
		cp += 2;
		cpp+= 2;
		ap += m*2;
		app+= m*2;
	    }
	}

    }

}


static void matmul_for_nk4_4(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=4;
    const int mm = 32;
    double btmp[m][m];
    v2df b2tmp[m][m];
    double atmp[m][mm];

    for(i=0;i<m;i++)for(j=0;j<m;j++)b2tmp[i][j]=(v2df){b[i][j],b[i][j]};
    //#pragma omp parallel for private(j,i,atmp)
    // use of OMP does not speed things up here...
    for(j=0;j<n;j+=mm){
	int jj;
	for(i=0;i<m;i++){
	    v2df * dest = (v2df*)(atmp[i]);
	    v2df * src = (v2df*)(a[i]+j);
#if 0	    
	    for(k=0;k<mm;k++)atmp[i][k]=a[i][j+k];
#endif
#if 0	    
	    for(k=0;k<mm/2;k++)dest[k]=src[k];
#endif
	    for(k=0;k<mm/2;k+=2){
		dest[k]=src[k];
		dest[k+1]=src[k+1];
	    }
	}
	for(i=0;i<m;i++){
	    int jjend = mm;
	    v2df * bp = b2tmp[i];
	    if (jjend+j > n) jjend = n-j;
	    for(jj=0;jj<jjend;jj+=2){
		v2df* cp = (v2df*)(&c[i][j+jj]);
		v2df* ap = (v2df*)(atmp[0]+jj);
		v2df* cpp = (v2df*)(&c[i][j+jj+2]);
		v2df* app = (v2df*)(atmp[0]+jj+2);
		*cp -= (*ap)*bp[0]
		    +(*(ap+16))*bp[1]
		    +(*(ap+32))*bp[2]
		    +(*(ap+48))*bp[3];
#if 0		
		*cpp -= (*app)*bp[0]
		    +(*(app+16))*bp[1]
		    +(*(app+32))*bp[2]
		    +(*(app+48))*bp[3];
#endif		
	    }
	}
    }

}
static void matmul_for_nk4_1(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i,j,k,l;
    const int m=4;
    const int mm = 32;
    double btmp[m][m];
    v2df b2tmp[m][m];

    for(i=0;i<m;i++)for(j=0;j<m;j++)b2tmp[i][j]=(v2df){b[i][j],b[i][j]};
#pragma omp parallel for private(j,i)
    // use of OMP does not speed things up here...
    for(j=0;j<n;j+=mm){
	double atmp[m][mm];
	int jj;
	for(i=0;i<m;i++){
	    v2df * dest = (v2df*)(atmp[i]);
	    v2df * src = (v2df*)(a[i]+j);
#if 0	    
	    for(k=0;k<mm;k++)atmp[i][k]=a[i][j+k];
#endif
#if 0	    
	    for(k=0;k<mm/2;k++)dest[k]=src[k];
#endif
	    for(k=0;k<mm/2;k+=4){
		dest[k]=src[k];
		dest[k+1]=src[k+1];
		dest[k+2]=src[k+2];
		dest[k+3]=src[k+3];
	    }
	}
	for(i=0;i<m;i++){
	    int jjend = mm;
	    v2df * bp = b2tmp[i];
	    if (jjend+j > n) jjend = n-j;
	    for(jj=0;jj<jjend;jj+=4){
		v2df* cp = (v2df*)(&c[i][j+jj]);
		v2df* ap = (v2df*)(atmp[0]+jj);
		v2df* cpp = (v2df*)(&c[i][j+jj+2]);
		v2df* app = (v2df*)(atmp[0]+jj+2);
		*cp -= (*ap)*bp[0]
		    +(*(ap+16))*bp[1]
		    +(*(ap+32))*bp[2]
		    +(*(ap+48))*bp[3];
		*cpp -= (*app)*bp[0]
		    +(*(app+16))*bp[1]
		    +(*(app+32))*bp[2]
		    +(*(app+48))*bp[3];
	    }
	}
    }

}
static void matmul_for_nk2_0(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int j,k,l;
    for(j=0;j<n;j++){
	    c[0][j] -= a[0][j]*b[0][0]+a[1][j]*b[0][1];
	    c[1][j] -= a[0][j]*b[1][0]+a[1][j]*b[1][1];
    }
}
static void matmul_for_nk2_1(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int j,k,l;

    for(j=0;j<n;j+=2){
	    c[0][j] -= a[0][j]*b[0][0]+a[1][j]*b[0][1];
	    c[1][j] -= a[0][j]*b[1][0]+a[1][j]*b[1][1];
	    c[0][j+1] -= a[0][j+1]*b[0][0]+a[1][j+1]*b[0][1];
	    c[1][j+1] -= a[0][j+1]*b[1][0]+a[1][j+1]*b[1][1];
    }
}
static void matmul_for_nk2_2(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int j,k,l;

    register double b00 = b[0][0];
    register double b01 = b[0][1];
    register double b10 = b[1][0];
    register double b11 = b[1][1];
    for(j=0;j<n;j+=2){
	    c[0][j] -= a[0][j]*b00+a[1][j]*b01;
	    c[0][j+1] -= a[0][j+1]*b00+a[1][j+1]*b01;
	    c[1][j] -= a[0][j]*b10+a[1][j]*b11;
	    c[1][j+1] -= a[0][j+1]*b10+a[1][j+1]*b11;
    }

}
static void matmul_for_nk2(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int j,k,l;
    register v2df b00 = (v2df){b[0][0],b[0][0]};
    register v2df b01 = (v2df){b[0][1],b[0][1]};
    register v2df b10 = (v2df){b[1][0],b[1][0]};
    register v2df b11 = (v2df){b[1][1],b[1][1]};
    v2df * a0 = (v2df*) a[0];
    v2df * a1 = (v2df*) a[1];
    v2df * c0 = (v2df*) c[0];
    v2df * c1 = (v2df*) c[1];
    int nh = n>>1;
    if (nh & 1){
	j=nh-1;
	c0[j] -= a0[j]*b00+a1[j]*b01;
	c1[j] -= a0[j]*b10+a1[j]*b11;
	nh = nh-1;
    }
    for(j=0;j<nh;j+=2){
	c0[j] -= a0[j]*b00+a1[j]*b01;
	c0[j+1] -= a0[j+1]*b00+a1[j+1]*b01;
	c1[j] -= a0[j]*b10+a1[j]*b11;
	c1[j+1] -= a0[j+1]*b10+a1[j+1]*b11;
	//	__builtin_prefetch((double*)&a0[j+32],0);
	//	__builtin_prefetch((double*)&a1[j+32],0);
	//	__builtin_prefetch((double*)&c0[j+32],1);
	//	__builtin_prefetch((double*)&c1[j+32],1);
	//	asm("prefetcht2 %0"::"m"(a0[j+32]):"memory");
	//	asm("prefetcht2 %0"::"m"(a1[j+32]):"memory");
    }
}


static void matmul_for_nk4_5(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{

    matmul_for_nk2(n1, a, n2, b, n3,c, n);
    matmul_for_nk2(n1, (double(*)[])  (a[2]), n2,
		   (double(*)[])  &(b[0][2]), n3,c, n);
    matmul_for_nk2(n1, a, n2, (double(*)[])  b[2], n3,
		   (double(*)[])  c[2], n);
    matmul_for_nk2(n1, (double(*)[])  &(a[2]), n2,
		   (double(*)[])  &(b[2][2]), n3,
		   (double(*)[])  c[2], n);
    

}

static void matmul_for_nk4_6(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    register v2df b00 = (v2df){b[0][0],b[0][0]};
    register v2df b01 = (v2df){b[0][1],b[0][1]};
    register v2df b02 = (v2df){b[0][2],b[0][2]};
    register v2df b03 = (v2df){b[0][3],b[0][3]};
    register v2df b10 = (v2df){b[1][0],b[1][0]};
    register v2df b11 = (v2df){b[1][1],b[1][1]};
    register v2df b12 = (v2df){b[1][2],b[1][2]};
    register v2df b13 = (v2df){b[1][3],b[1][3]};
    register v2df b20 = (v2df){b[2][0],b[2][0]};
    register v2df b21 = (v2df){b[2][1],b[2][1]};
    register v2df b22 = (v2df){b[2][2],b[2][2]};
    register v2df b23 = (v2df){b[2][3],b[2][3]};
    register v2df b30 = (v2df){b[3][0],b[3][0]};
    register v2df b31 = (v2df){b[3][1],b[3][1]};
    register v2df b32 = (v2df){b[3][2],b[3][2]};
    register v2df b33 = (v2df){b[3][3],b[3][3]};

    v2df * a0 = (v2df*) a[0];
    v2df * a1 = (v2df*) a[1];
    v2df * a2 = (v2df*) a[2];
    v2df * a3 = (v2df*) a[3];
    v2df * c0 = (v2df*) c[0];
    v2df * c1 = (v2df*) c[1];
    v2df * c2 = (v2df*) c[2];
    v2df * c3 = (v2df*) c[3];
    int nh = n>>1;
    //#pragma omp parallel
    {
	//#pragma omp section
	{
	    int j;
	    //#pragma omp for private (j)
	    for(j=0;j<nh;j++){
		c0[j] -= a0[j]*b00+a1[j]*b01+a2[j]*b02+a3[j]*b03;
		c1[j] -= a0[j]*b10+a1[j]*b11+a2[j]*b12+a3[j]*b13;
	    }
	}
	//#pragma omp section
	{
	    int j;
	    //#pragma omp for private (j)
	    for(j=0;j<nh;j++){
		c2[j] -= a0[j]*b20+a1[j]*b21+a2[j]*b22+a3[j]*b23;
		c3[j] -= a0[j]*b30+a1[j]*b31+a2[j]*b32+a3[j]*b33;
	    }
	}
    }
}
static void matmul_for_nk4_7(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    v2df * a0 = (v2df*) a[0];
    v2df * a1 = (v2df*) a[1];
    v2df * a2 = (v2df*) a[2];
    v2df * a3 = (v2df*) a[3];
    int j;
    {
	int nh = n>>1;
	v2df * c0 = (v2df*) c[0];
	v2df * c1 = (v2df*) c[1];
	 v2df b00 = (v2df){b[0][0],b[0][0]};
	 v2df b01 = (v2df){b[0][1],b[0][1]};
	 v2df b02 = (v2df){b[0][2],b[0][2]};
	 v2df b03 = (v2df){b[0][3],b[0][3]};
	 v2df b10 = (v2df){b[1][0],b[1][0]};
	 v2df b11 = (v2df){b[1][1],b[1][1]};
	 v2df b12 = (v2df){b[1][2],b[1][2]};
	 v2df b13 = (v2df){b[1][3],b[1][3]};
	if (nh & 1){
	    j=nh-1;
	    c0[j] -= a0[j]*b00+a1[j]*b01+a2[j]*b02+a3[j]*b03;
	    c1[j] -= a0[j]*b10+a1[j]*b11+a2[j]*b12+a3[j]*b13;
	    nh--;
	}
	for(j=0;j<nh;j+=2){
	    c0[j] -= a0[j]*b00+a1[j]*b01+a2[j]*b02+a3[j]*b03;
	    c1[j] -= a0[j]*b10+a1[j]*b11+a2[j]*b12+a3[j]*b13;
	    c0[j+1] -= a0[j+1]*b00+a1[j+1]*b01+a2[j+1]*b02+a3[j+1]*b03;
	    c1[j+1] -= a0[j+1]*b10+a1[j+1]*b11+a2[j+1]*b12+a3[j+1]*b13;
	}
    }
    {
	int nh = n>>1;
	v2df * c2 = (v2df*) c[2];
	v2df * c3 = (v2df*) c[3];
	 v2df b20 = (v2df){b[2][0],b[2][0]};
	 v2df b21 = (v2df){b[2][1],b[2][1]};
	 v2df b22 = (v2df){b[2][2],b[2][2]};
	 v2df b23 = (v2df){b[2][3],b[2][3]};
	register v2df b30 = (v2df){b[3][0],b[3][0]};
	register v2df b31 = (v2df){b[3][1],b[3][1]};
	register v2df b32 = (v2df){b[3][2],b[3][2]};
	register v2df b33 = (v2df){b[3][3],b[3][3]};
	
	for(j=0;j<nh;j++){
	    c2[j] -= a0[j]*b20+a1[j]*b21+a2[j]*b22+a3[j]*b23;
	    c3[j] -= a0[j]*b30+a1[j]*b31+a2[j]*b32+a3[j]*b33;
	}
    }
}


static void matmul_for_nk4_8(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    v2df * a0 = (v2df*) a[0];
    v2df * a1 = (v2df*) a[1];
    v2df * a2 = (v2df*) a[2];
    v2df * a3 = (v2df*) a[3];
    int nh = n>>1;
    int j;

    int k;
    v2df * cv;
    v2df * cvv;
    register v2df b0;
    register v2df b1;
    register v2df b2;
    register v2df b3;
    register v2df b4;
    register v2df b5;
    register v2df b6;
    register v2df b7;
    for(k=0;k<4;k+=2){
	cv = (v2df*) c[k];
	cvv = (v2df*) c[k+1];
	b0 = (v2df){b[k][0],b[k][0]};
	b1 = (v2df){b[k][1],b[k][1]};
	b2 = (v2df){b[k][2],b[k][2]};
	b3 = (v2df){b[k][3],b[k][3]};
	b4 = (v2df){b[k+1][0],b[k+1][0]};
	b5 = (v2df){b[k+1][1],b[k+1][1]};
	b6 = (v2df){b[k+1][2],b[k+1][2]};
	b7 = (v2df){b[k+1][3],b[k+1][3]};
	for(j=0;j<nh;j++){
	    register v2df aa0 = a0[j];
	    register v2df aa1 = a1[j];
	    register v2df aa2 = a2[j];
	    register v2df aa3 = a3[j];
	    register v2df x = aa0*b0;
	    x+= aa1*b1;
	    x+= aa2*b2;
	    x+= aa3*b3;
	    cv[j]-=x;
	    x = aa0*b4;
	    x+= aa1*b5;
	    x+= aa2*b6;
	    x+= aa3*b7;
	    cvv[j] -= x;
	}
    }
}

static void matmul_for_nk4(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int m0,m1,m2,m3;
    int s1, s2, s3;
    int ds = (n/32)*8;
    s1 = ds;
    s2 = ds*2;
    s3 = ds*3;
    m0 = ds;
    m1 = ds;
    m2 = ds;
    m3 = n-s3;
    //    fprintf(stderr,"n, s, m = %d %d %d %d %d %d %d %d\n",
    //    n,s1,s2,s3,m0,m1,m2,m3);
    
#pragma omp parallel
#pragma omp sections
  {
#pragma omp section
      matmul_for_nk4_7(n1,a,n2,b,n3,c,m0);
#pragma omp section
      matmul_for_nk4_7(n1,((double*)a)+s1 ,n2,b,n3,((double*)c)+s1,m1);
#pragma omp section
      matmul_for_nk4_7(n1,((double*)a)+s2 ,n2,b,n3,((double*)c)+s2,m2);
#pragma omp section
      matmul_for_nk4_7(n1,((double*)a)+s3 ,n2,b,n3,((double*)c)+s3,m3);
  }
}
static void matmul_for_nk8_9(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    matmul_for_nk4(n1, a, n2, b, n3,c, n);
    matmul_for_nk4(n1, (double(*)[])  (a[4]), n2,
		   (double(*)[])  &(b[0][4]), n3,c, n);
    matmul_for_nk4(n1, a, n2, (double(*)[])  b[4], n3,
		   (double(*)[])  c[4], n);
    matmul_for_nk4(n1, (double(*)[])  &(a[4]), n2,
		   (double(*)[])  &(b[4][4]), n3,
		   (double(*)[])  c[4], n);
    
}
#if 0
static void matmul_for_nk8(int n1, double a[][n1],
			   int n2, double b[][n2],
			   int n3, double c[][n3],
			   int n)
{
    int i;
    int nb=384;
    for (i=0;i<n;i+=nb){
	int iend = i+nb;
	if (iend > n) iend = n;
	matmul_for_nk8_worker(n1, (double(*)[]) (a[0]+i),
			      n2,  (double(*)[]) (b[0]+i),
			      n3, (double(*)[]) (c[0]+i),
			      iend-i);
    }
}
#endif

static void matmul_for_small_nk_local(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{
    // simplest version
    int j,k,l;
    BEGIN_TSC;
    if (kk == 2){
	matmul_for_nk2(n1, a, n2, b, n3,c, n);
	END_TSC(t,16);
	return;
    }
    if (kk == 4){
	matmul_for_nk4(n1, a, n2, b, n3,c, n);
	END_TSC(t,13);
	return;
    }
    if (kk == 8){
	matmul_for_nk8(n1, a, n2, b, n3,c, n);
	END_TSC(t,12);
	return;
    }
	
    for(j=0;j<n;j++)
	for(k=0;k<m;k++)
	    for(l=0;l<kk;l++)
		c[k][j] -= a[l][j]*b[k][l];
}

static void mmmulandsub(int n, double a[n+1][n], int rshift, int m0, int m1,
		 int c0, int c1, int r0,int r1)
{
    int j,k,l;
#ifdef TIMETEST
    BEGIN_TSC;
#endif
#ifndef NOBLAS
    if ((m1-m0)<=8){
	// =4 is slighly faster than =8 on Ci7 
	matmul_for_small_nk_local(n, (double(*)[])  &(a[m0-rshift][r0]),
			      n, (double(*)[])  &(a[c0-rshift][m0]),
			      n, (double(*)[])  &(a[c0-rshift][r0]),
			      c1-c0,m1-m0,r1-r0);
    }else{
	cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,
		     r1-r0, c1-c0, m1-m0, -1.0, &(a[m0-rshift][r0]), n,
		     &(a[c0-rshift][m0]), n, 1, &(a[c0-rshift][r0]), n );
    }
    // example:
    // r0, m0 = i+m,i
    // m0, c0 = i, i+m
    // r0, c0 = i+m, i+m
    //r1-r0 = n-i-m
    // c1-c0 = iend-i-m
    // m1-m0 = m
#else
    for(j=r0;j<r1;j++)
	for (k=c0;k<c1;k++)
	    for (l=m0; l<m1; l++)
		a[k-rshift][j] -= a[l-rshift][j]*a[k-rshift][l];
#endif    
#ifdef TIMETEST
    END_TSC(t,4);
#endif
}

static int nswap;
static void column_decomposition(int n, double a[][n],  int m, int pv[], int i)
{
    // shift a so that partial array is okay
    int  j, k;
    int ip,ii;
    double ainv;
#ifdef TIMETEST
    BEGIN_TSC;
#endif
    for(ip=0;ip<m;ip++){
	ii=i+ip;
	int p = findpivot(n,a,ii);
	pv[ip]=p;
	swaprows(n,a,p,ii,0,m);
	nswap++;
	// normalize row ii
	ainv = 1.0/a[ip][ii];
	scalerow(n,a,ainv,ii,0,ip);
	scalerow(n,a,ainv,ii,ip+1,m);
	// subtract row ii from all lower rows
	vvmulandsub(n,  a, ip,ii, ip+1, m, ii+1, n);
    }
#ifdef TIMETEST
    END_TSC(t,5);
#endif    
}	

static void process_right_part(int n,
			double a[n+1][n],
			int m,
			int pv[],
			int i,
			int iend)
{
    int ii;
    // exchange rows 
    for(ii=i;ii<i+m;ii++){
	swaprows(n,a,pv[ii-i],ii,m,iend-i);
    }
    
    // normalize rows
    for(ii=i;ii<i+m;ii++){
	scalerow(n,a,1.0/a[ii-i][ii] ,ii,m,iend-i);
    }
    // subtract rows (within i-i+m-1)
    for(ii=i;ii<i+m;ii++){
	vvmulandsub(n,  a, ii-i,ii,  m, iend-i, ii+1, i+m);
    }
    
    // subtract rows i-i+m-1 from all lower rows
    mmmulandsub(n, a, i, i,i+m, i+m, iend, i+m, n);
    //    fprintf(stderr,"process_r, end\n");
    //    usleep(1);
}
static void column_decomposition_recursive(int n,
				    double a[n+1][n],
				    int m,
				    int pv[],
				    int i)
{
    int  j, k;
    int ip,ii;
    double ainv;
    //    fprintf(stderr,"enter t column recursive %d %d\n", i, m);
    if (m <= 2){
	// perform non-recursive direct decomposition
	column_decomposition(n, a, m, pv,i);
    }else{	
	// process the left half by recursion
	//	fprintf(stderr,"call column recursive %d %d\n", i, m);
	column_decomposition_recursive(n, a, m/2, pv,i);
	// process the right half
	//	fprintf(stderr,"call right part %d %d\n", i, m);
	process_right_part(n,a,m/2,pv,i,i+m);
	//	fprintf(stderr,"call right recursive %d %d\n", i, m);
	column_decomposition_recursive(n, a+m/2, m/2, pv+m/2,i+m/2);
	// process the swap of rows for the left half
	//	fprintf(stderr,"call swaprowse %d %d\n", i, m);
	for(ii=i+m/2;ii<i+m;ii++){
	    swaprows(n,a,pv[ii-i],ii,0,m/2);
	}
	//	fprintf(stderr,"call scalerows %d %d\n", i, m);
	// normalize rows
	for(ii=i+m/2;ii<i+m;ii++){
	    scalerow(n,a,1.0/a[ii-i][ii] ,ii,0,m/2);
	}
    }
}
    
void cm_column_decomposition_recursive(int n,
				    double a[n+1][n],
				    int m,
				    int pv[],
				    int i)
{
    column_decomposition_recursive( n, a, m, pv, i);
}
void cm_column_decomposition(int n,
				    double a[n+1][n],
				    int m,
				    int pv[],
				    int i)
{
    column_decomposition( n, a, m, pv, i);
}
void cm_process_right_part(int n,
			   double a[n+1][n],
			   int m,
			   int pv[],
			   int i,
			   int iend)
{
    process_right_part(n, a, m, pv, i,iend);
}

