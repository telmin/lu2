// lu2.c
//
// test program for blocked LU decomposition
//
// Time-stamp: <11/05/05 16:59:38 makino>
//#define NOBLAS
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>

#include <emmintrin.h>
typedef double v2df __attribute__((vector_size(16)));
typedef union {v2df v; double s[2];}v2u;

#include <lu2tlib.h>
#include <lu2lib.h>
void timer_init();
double cpusec();
double wsec();


#define RDIM (n+16)
void copymats( int n, double a[n][RDIM], double a2[n][RDIM])
	      
{
    int i, j;
    for(i=0;i<n;i++){
	for(j=0;j<n+2;j++) a2[i][j] = a[i][j];
    }
}

void copybvect( int n, double a[][RDIM], double b[])
	      
{
    int i;
    for(i=0;i<n;i++)b[i] = a[i][n];
}

void showresult(int n, double a[n][RDIM], double x[])
{
    int i, j;
    double emax = 0;
    for(i=0;i<n;i++){
	int k;
	double b2=0;
	//	printf("%3d: ", i);
	//	for(j=0;j<n;j++) printf(" %10.3e", a[i][j]);
	for(j=0;j<n;j++) b2 += a[i][j] * x[j];
	double err = b2-a[i][n];
        emax =  (fabs(err) > emax) ? fabs(err):emax;
	//	printf(" %10.3e  %10.3e %10.3e %10.3e \n", x[i], a[i][n], b2, err);
    }
    printf("Emax= %10.3e\n", emax);
}
    
    
void readmat( int n, double a[n][RDIM])
{
    int i, j;
    for(i=0;i<n;i++){
	for(j=0;j<n+1;j++) scanf("%le", &(a[i][j]));
    }
}
void randomsetmat( int n, int seed, double a[n][RDIM])
{
    long int i, j;
    srand48((long) seed);
    for(i=0;i<n;i++){
	//	printf("i=%d\n", i);
	double * ap = a[i];
	for(j=0;j<n;j++) {
	    //		    ap[j]=drand48();
	    ap[j]=drand48()-0.5;
	}
	//	printf("n, i=%d\n", i);
	//	a[i][n]=1;
	a[i][n]=drand48()-0.5;
    }
}

void printmat( int n, double a[n][RDIM])

{
    int i, j;
    for(i=0;i<n;i++){
	printf("%3d: ", i);
	for(j=0;j<n+1;j++) printf(" %10.3e", a[i][j]);
	printf("\n");
    }
    printf("\n");
}
void printsqmat( int n, double a[n][n])

{
    int i, j;
    for(i=0;i<n;i++){
	printf("%3d: ", i);
	for(j=0;j<n;j++) printf(" %10.3e", a[i][j]);
	printf("\n");
    }
    printf("\n");
}

void backward_sub(int n,double a[n][RDIM], double b[])
{
    int i,j,k;
    for (i=0;i<n;i++)b[i] = a[i][n];
    for(j=n-2;j>=0;j--)
	for(k=j+1;k<n;k++) b[j] -= b[k]*a[j][k];
}

    
void lu( int n, double a[n][RDIM], double b[])
{
    int i, j, k;
    for(i=0;i<n-1;i++){
	// select pivot
	double amax = fabs(a[i][i]);
	int p=i;
	for(j=i+1;j<n;j++){
	    if (fabs(a[j][i]) > amax){
		amax = fabs(a[j][i]);
		p = j;
	    }
	}
	// exchange rows
	if (p != i){
	    for(j=i;j<n+1;j++){
		double tmp = a[p][j];
		a[p][j] = a[i][j];
		a[i][j]=tmp;
	    }
	}
		
	// normalize row i
	double ainv = 1.0/a[i][i];
	//	fprintf(stderr,"%d %e\n", i, ainv);
	for(k=i+1;k<n+1;k++) a[i][k]*= ainv;
	// subtract row i from all lower rows
	for(j=i+1;j<n;j++){
	    //	    fprintf(stderr,"j=%d \n",j);
	    for(k=i+1;k<n+1;k++) a[j][k] -= a[j][i] * a[i][k];
	}
    }
    printmat(n,a);
    
    a[n-1][n] /= a[n-1][n-1];
    backward_sub(n,a,b);
}



int findpivot(int n, double a[n][RDIM], int current)
{
    double amax = fabs(a[current][current]);
    int i;
    int p=current;
    for(i=current+1;i<n;i++){
	if (fabs(a[i][current]) > amax){
	    amax = fabs(a[i][current]);
	    p = i;
	}
    }
    return p;
}

void scalerow( int n, double a[n][RDIM], double scale,
	       int row, int cstart, int cend)
{
    int j;
    BEGIN_TSC;
    int jmax = (cend+1-cstart)/2;
    v2df *a1 = (v2df*)(a[row]+cstart);
    v2df ss = (v2df){scale,scale};
    for(j=0;j<jmax;j+=2){
	__builtin_prefetch(a1+j+16,1,0);
	a1[j] *= ss;
	a1[j+1]*= ss;
    }
    END_TSC(t,1);
}


void swaprows(int n, double a[n][RDIM], int row1, int row2,
	      int cstart, int cend)
{
    /* WARNING: works only for row1 % 4 = 0 and RDIM >= n+4*/
    
    int j;
    if (row1 != row2){
	int jmax = (cend+1-cstart)/2;
#ifdef TIMETEST
    BEGIN_TSC;
#endif
	v2df *a1, *a2, tmp, tmp1;
	a1 = (v2df*)(a[row1]+cstart);
	a2 = (v2df*)(a[row2]+cstart);
	for(j=0;j<jmax;j+=2){
	    tmp = a1[j];
	    tmp1 = a1[j+1];
	    a1[j]=a2[j];
	    a1[j+1]=a2[j+1];
	    a2[j]=tmp;
	    a2[j+1]=tmp1;
	    __builtin_prefetch(a1+j+16,1,0);
	    __builtin_prefetch(a2+j+16,1,0);
	    // prefetch options: 1: for write, 0: read only
	    // 0: need not be kept in cache
	    // 3: should be there for as long as possible
	    
	}
#ifdef TIMETEST
	END_TSC(t,0);
#endif    
    }
}
void swaprows_simple(int n, double a[n][RDIM], int row1, int row2,
	      int cstart, int cend)
{
    /* WARNING: works only for row1 % 4 = 0 and RDIM >= n+4*/
    
    int j;
    if (row1 != row2){
	int jmax = (cend+1-cstart)/2;
#if 1
	v2df *a1, *a2, tmp, tmp1;
	a1 = (v2df*)(a[row1]+cstart);
	a2 = (v2df*)(a[row2]+cstart);
	for(j=0;j<jmax;j++){
	    tmp = a1[j];
	    a1[j]=a2[j];
	    a2[j]=tmp;
	}
#endif
#if 0	
	for(j=cstart;j<cend;j++){
	    double tmp = a[row1][j];
	    a[row1][j]=a[row2][j];
	    a[row2][j]=tmp;
	}
#endif	
    }
}

void swaprows_simple_with_scale(int n, double a[n][RDIM], int row1, int row2,
		     int cstart, int cend, double scale)
{
    /* WARNING: works only for row1 % 4 = 0 and RDIM >= n+4*/
    
    int j;
    if (row1 != row2){
	int jmax = (cend+1-cstart)/2;
#if 1
	v2df *a1, *a2, tmp, tmp1;
	v2df ss = (v2df){scale,scale};
	a1 = (v2df*)(a[row1]+cstart);
	a2 = (v2df*)(a[row2]+cstart);
	for(j=0;j<(jmax & (0xfffffffe));j+=2){
	    __builtin_prefetch(a1+j+32,1,0);
	    __builtin_prefetch(a2+j+32,1,0);
	    tmp = a1[j];
	    a1[j]=a2[j];
	    a2[j]=tmp*ss;
	    tmp1 = a1[j+1];
	    a1[j+1]=a2[j+1];
	    a2[j+1]=tmp1*ss;

	}
	if (jmax &1){
	    tmp = a1[jmax-1];
	    a1[jmax-1]=a2[jmax-1];
	    a2[jmax-1]=tmp*ss;
	}	    
#endif
#if 0	
	for(j=cstart;j<cend;j++){
	    double tmp = a[row1][j];
	    a[row1][j]=a[row2][j];
	    a[row2][j]=tmp;
	}
#endif	
    }else{
	scalerow(n,a,scale ,row2,cstart,cend);
    }
}



void swapelements(double b[], int i1, int i2)
{
    double tmp;
    tmp=b[i1]; b[i1]=b[i2]; b[i2]=tmp;
}
			   

void vvmulandsub(int n, double a[n][RDIM], int current,
		 int c0, int c1, int r0,int r1)
{
    int j,k;
    for(j=r0;j<r1;j++)
	for (k=c0;k<c1;k++)
	    a[j][k] -= a[j][current]*a[current][k];
}

	    
void mmmulandsub(int n, double a[n][RDIM], int m0, int m1,
		 int c0, int c1, int r0,int r1)
{
    int j,k,l;
    if (c1-c0 <16){
	int np=n+1;
	matmul_for_small_nk(RDIM, (double(*)[]) (&a[r0][m0]),
			    RDIM, (double(*)[]) (&a[m0][c0]),
			    RDIM, (double(*)[]) (&a[r0][c0]),
			    r1-r0,
			    m1-m0,
			    c1-c0);
    }else{	
#ifndef NOBLAS
	
	mydgemm(r1-r0, c1-c0, m1-m0, -1.0, &(a[r0][m0]), RDIM,
		&(a[m0][c0]), RDIM, 1.0, &(a[r0][c0]), RDIM );
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
		    a[j][k] -= a[j][l]*a[l][k];
#endif
    }
}

static int nswap;
void column_decomposition(int n, double a[n][RDIM],  int m, int pv[], int i)
{
    int  j, k;
    int ip,ii;
    double ainv;
    for(ip=0;ip<m;ip++){
	ii=i+ip;
	int p = findpivot(n,a,ii);
	if (fabs(a[p][ii]) > 2* fabs(a[ii][ii])){
	    pv[ip]=p;
	    swaprows(n,a,p,ii,i,i+m);
	    nswap++;
	}else{
	    pv[ip]=ii;
	}
	// normalize row ii
	ainv = 1.0/a[ii][ii];
	scalerow(n,a,ainv,ii,i,ii);
	scalerow(n,a,ainv,ii,ii+1,i+m);
	// subtract row ii from all lower rows
	vvmulandsub(n,  a, ii, ii+1, i+m, ii+1, n);
    }
}	

static void solve_triangle_for_unit_mat_internal(int n,
				 double a[][RDIM],
				 int nb,
				 double b[][nb],
				 int m)
{
    int ii,j,k;
    for(ii=0;ii<m;ii++)
	for(j=ii+1;j<m;j++)
	    for (k=0;k<m;k++)
		b[j][k] -= a[j][ii]*b[ii][k];
}    

void solve_triangle_for_unit_mat(int n,
				 double a[n][RDIM],
				 int nb,
				 double b[nb][nb],
				 int m,
				 int i);

static void solve_triangle_for_unit_mat_recursive(int n,
				 double a[][RDIM],
				 int nb,
				 double b[][nb],
						    int m);
static void solve_triangle_for_unit_mat_recursive_0(int n,
						    double a[][RDIM],
						    int nb,
						    double b[][nb],
						    int m)
{
    int i,ii,j,k;
    if (m < 16){
	solve_triangle_for_unit_mat_internal(n, a, nb, b,m);
	return;
    }
    const int mhalf = m/2;
    solve_triangle_for_unit_mat_recursive(n, a, nb, b,mhalf);
    
    mydgemm( mhalf, mhalf, mhalf, -1.0, &(a[mhalf][0]), RDIM,
		 &(b[0][0]), nb, 1.0, &(b[mhalf][0]),nb );

    double bwork[mhalf][mhalf];
    double bwork2[mhalf][mhalf];
    for (j=0;j<mhalf;j++)
	for (k=0;k<mhalf;k++)bwork[j][k]=0.0;
    for (j=0;j<mhalf;j++)bwork[j][j]=1.0;
    solve_triangle_for_unit_mat_recursive(n, (double(*)[])(&a[mhalf][mhalf]),
				mhalf, bwork,mhalf);
    for(i=0;i<mhalf;i++)
	for(j=0;j<mhalf;j++)
	    bwork2[i][j]=b[i+mhalf][j];
    mydgemm(mhalf, mhalf, mhalf, 1.0, (double*)bwork,mhalf,
		 (double*)bwork2, mhalf, 0.0, &(b[mhalf][0]),nb );

    solve_triangle_for_unit_mat_recursive(n, (double(*)[])(&a[mhalf][mhalf]),
					 nb, (double(*)[])(&b[mhalf][mhalf]),
					 mhalf);

}    
static void solve_triangle_for_unit_mat_recursive(int n,
				 double a[][RDIM],
				 int nb,
				 double b[][nb],
				 int m)
{
    int i,ii,j,k;
    if (m < 16){
	// apparently, too deep recursion here
	// causes large error....
	// might need some fix
	
	solve_triangle_for_unit_mat_internal(n, a, nb, b,m);
	return;
    }
    const int mhalf = m/2;
    solve_triangle_for_unit_mat_recursive(n, a, nb, b,mhalf);
    
    mydgemm( mhalf, mhalf, mhalf, -1.0, &(a[mhalf][0]), RDIM,
		 &(b[0][0]), nb, 1.0, &(b[mhalf][0]),nb );

    double bwork[mhalf][mhalf];
    double bwork2[mhalf][mhalf];
    for (j=0;j<mhalf;j++)
	for (k=0;k<mhalf;k++)bwork[j][k]=0.0;
    for (j=0;j<mhalf;j++)bwork[j][j]=1.0;
    solve_triangle_for_unit_mat_recursive(n, (double(*)[])(&a[mhalf][mhalf]),
				mhalf, bwork,mhalf);
    for(i=0;i<mhalf;i++)
	for(j=0;j<mhalf;j++)
	    bwork2[i][j]=b[i+mhalf][j];
    mydgemm(mhalf, mhalf, mhalf, 1.0, (double*)bwork,mhalf,
		 (double*)bwork2, mhalf, 0.0, &(b[mhalf][0]),nb );
    for (j=0;j<mhalf;j++)
	for (k=0;k<j+1;k++)b[mhalf+j][mhalf+k]=bwork[j][k];
}    

void solve_triangle_for_unit_mat(int n,
				 double a[n][RDIM],
				 int nb,
				 double b[nb][nb],
				 int m,
				 int i)
{
    int ii,j,k;
    BEGIN_TSC;
    for (j=0;j<nb;j++)
	for (k=0;k<nb;k++)b[j][k]=0.0;
    for (j=0;j<nb;j++)b[j][j]=1.0;
	
    solve_triangle_for_unit_mat_recursive(n, (double(*)[])  (&a[i][i]),
					 nb, b, m);
    END_TSC(t,8);
}    

void solve_triangle(int n,
		    double a[n][RDIM],
		    int m,
		    double awork[][n],
		    int i,
		    int iend)
{
    int ii,j,k;
    //    current =ii
    //   c0=i+m
    //   c1=iend
    //   r0=ii+1
    //   r1 = i+m
    double b[m][m];
    double work[m];
    solve_triangle_for_unit_mat(n,a,m,b,m,i);
    BEGIN_TSC;
    for(j=i;j<i+m;j++){
	for (k=i+m;k<iend;k++){
	    awork[j-i][k-i-m]=a[j][k];
#ifdef NOBLAS	    
	    a[j][k]=0;
#endif	    
	}
    }
#ifndef NOBLAS
    mydgemm(m, iend-i-m, m, 1.0, &(b[0][0]), m,
		 &(awork[0][0]), n, 0.0, &(a[i][i+m]), RDIM );
#else
    for (k=i+m;k<iend;k++){
	for(j=0;j<m;j++)
	    for(ii=0;ii<j+1;ii++)
		a[j+i][k]+=  b[j][ii]*awork[ii][k-i-m];
    }
#endif    
    END_TSC(t,9);
}    


void process_right_part(int n,
			double a[n][RDIM],
			int m,
			double awork[][n],
			int pv[],
			int i,
			int iend)
{
    int ii;
    // exchange rows
    if ((iend-i-m) > m){
	//    if(0){
	int k;
        int nt=4;
#ifdef TIMETEST
    BEGIN_TSC;
#endif
#pragma omp parallel for private(k,ii)
	for(k=0;k<nt;k++){
	    int di = (16+iend-i-m)/nt;
	    int istart = i+m+di*k;
	    int iend2 = istart + di;
	    if (iend2> iend) iend2 = iend;
	    //	    fprintf(stderr," swaprows %d %d %d %d\n",istart,iend2,i+m,iend);
	    for(ii=i;ii<i+m;ii++){
		//		swaprows_simple(n,a,pv[ii-i],ii,istart,iend2);
		//		scalerow(n,a,1.0/a[ii][ii] ,ii,istart,iend2);
		swaprows_simple_with_scale(n,a,pv[ii-i],ii,istart,iend2,
					   1.0/a[ii][ii] );
	    }
	}
	// normalize rows
#ifdef TIMETEST
	END_TSC(t,0);
#endif    
#pragma omp parallel for private(ii)
	for(ii=i;ii<i+m;ii++){
	    //	    scalerow(n,a,1.0/a[ii][ii] ,ii,i+m,iend);
	}
    }else{
	for(ii=i;ii<i+m;ii++){
	    swaprows(n,a,pv[ii-i],ii,i+m,iend);
	    scalerow(n,a,1.0/a[ii][ii] ,ii,i+m,iend);
	}
	
	// normalize rows
	for(ii=i;ii<i+m;ii++){
	}
    }	
    // subtract rows (within i-i+m-1)
    solve_triangle(n,a,m,awork, i,iend);
    //    for(ii=i;ii<i+m;ii++){
    //	vvmulandsub(n,  a, ii,  i+m, iend, ii+1, i+m);
    //    }
    
    // subtract rows i-i+m-1 from all lower rows
    mmmulandsub(n, a, i,i+m, i+m, iend, i+m, n);
}

void transpose_rowtocol8(int n, double a[][RDIM], double at[][n],
			 int istart)
{
    int i,j,k;
    const int m=8;
    double atmp[m][m] __attribute__((align(128)));
#pragma omp parallel for private(i,j,k,atmp)
    for(i=istart;i<n;i+=m){
	for(k=0;k<m;k++){
	    double *ak = a[i+k];
	    atmp[0][k]  =ak[0];
	    atmp[1][k]  =ak[1];
	    atmp[2][k]  =ak[2];
	    atmp[3][k]  =ak[3];
	    atmp[4][k]  =ak[4];
	    atmp[5][k]  =ak[5];
	    atmp[6][k]  =ak[6];
	    atmp[7][k]  =ak[7];
	}
	for(j=0;j<m;j++){
	    v2df * atp = (v2df*) atmp[j];
	    v2df * ap = (v2df*) (at[j]+i);
	    *(ap)=*(atp);
	    *(ap+1)=*(atp+1);
	    *(ap+2)=*(atp+2);
	    *(ap+3)=*(atp+3);
	}
    }
}
void transpose_rowtocol16_0(int n, double a[][RDIM], double at[][n],
			 int istart)
{
    int i,j,k;
    const int m=16;
    const int m4=16;
    double atmp[m][m4];
    int mend;
    
#pragma omp parallel for private(i,j,k,atmp)
    for(i=istart;i<n;i+=m4){
	mend = m4;
	if (mend+i > n) mend = n-i;
	for(k=0;k<mend;k++){
	    double *ak = a[i+k];
	    //	    __builtin_prefetch(a+i+k+m,0,0);
	    atmp[0][k]  =ak[0];
	    atmp[1][k]  =ak[1];
	    atmp[2][k]  =ak[2];
	    atmp[3][k]  =ak[3];
	    atmp[4][k]  =ak[4];
	    atmp[5][k]  =ak[5];
	    atmp[6][k]  =ak[6];
	    atmp[7][k]  =ak[7];
	    atmp[8][k]  =ak[8];
	    atmp[9][k]  =ak[9];
	    atmp[10][k]  =ak[10];
	    atmp[11][k]  =ak[11];
	    atmp[12][k]  =ak[12];
	    atmp[13][k]  =ak[13];
	    atmp[14][k]  =ak[14];
	    atmp[15][k]  =ak[15];
	}
	for(j=0;j<mend;j++){
	    v2df * atp = (v2df*) atmp[j];
	    v2df * ap = (v2df*) (at[j]+i);
	    *(ap)=*(atp);
	    *(ap+1)=*(atp+1);
	    *(ap+2)=*(atp+2);
	    *(ap+3)=*(atp+3);
	    *(ap+4)=*(atp+4);
	    *(ap+5)=*(atp+5);
	    *(ap+6)=*(atp+6);
	    *(ap+7)=*(atp+7);
	}
    }
}
void transpose_rowtocol16_1(int n, double a[][RDIM], double at[][n],
			 int istart)
{
    int i,j,k;
    const int m=16;
    double atmp[m][m];
    int mend;
    //#pragma omp parallel for private(i,j,k,atmp)
    for(i=istart;i<n;i+=m){
	for(k=0;k<m;k++){
	    v2df * ak = (v2df*) a[i+k];
	    v2df * akk = (v2df*) atmp[k];
	    akk[0]  =ak[0];
	    akk[1]  =ak[1];
	    akk[2]  =ak[2];
	    akk[3]  =ak[3];
	    akk[4]  =ak[4];
	    akk[5]  =ak[5];
	    akk[6]  =ak[6];
	    akk[7]  =ak[7];
	}
	for(j=0;j<m;j++){
	    v2df * atk= (v2df*)(at[j]+i);
	    atk[0]=(v2df){atmp[0][j],atmp[1][j]};
	    atk[1]=(v2df){atmp[2][j],atmp[3][j]};
	    atk[2]=(v2df){atmp[4][j],atmp[5][j]};
	    atk[3]=(v2df){atmp[6][j],atmp[7][j]};
	    atk[4]=(v2df){atmp[8][j],atmp[9][j]};
	    atk[5]=(v2df){atmp[10][j],atmp[11][j]};
	    atk[6]=(v2df){atmp[12][j],atmp[13][j]};
	    atk[7]=(v2df){atmp[14][j],atmp[15][j]};
	}
    }

}

void transpose_rowtocol16(int n, double a[][RDIM], double at[][n],
			 int istart)
{
    int i,j,k;
    const int m=16;
    int mend;
#pragma omp parallel for private(i,j,k)
    for(i=istart;i<n;i+=m){
	double atmp[m][m];
	//	BEGIN_TSC;
	for(k=0;k<m;k++){
	    v2df * ak = (v2df*) a[i+k];
	    v2df * akk = (v2df*) atmp[k];
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][0]):"memory");
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][8]):"memory");
	    //	    __builtin_prefetch(a[i+k+m*2],0,0);
	    //	    __builtin_prefetch(a[i+k+m*2]+8,0,0);
	    akk[0]  =ak[0];
	    akk[1]  =ak[1];
	    akk[2]  =ak[2];
	    akk[3]  =ak[3];
	    akk[4]  =ak[4];
	    akk[5]  =ak[5];
	    akk[6]  =ak[6];
	    akk[7]  =ak[7];
	}
	//	END_TSC(t,17);
	//	{
	//	BEGIN_TSC;
	for(j=0;j<m;j++){
	    v2df * atk= (v2df*)(at[j]+i);
	    atk[0]=(v2df){atmp[0][j],atmp[1][j]};
	    atk[1]=(v2df){atmp[2][j],atmp[3][j]};
	    atk[2]=(v2df){atmp[4][j],atmp[5][j]};
	    atk[3]=(v2df){atmp[6][j],atmp[7][j]};
	    atk[4]=(v2df){atmp[8][j],atmp[9][j]};
	    atk[5]=(v2df){atmp[10][j],atmp[11][j]};
	    atk[6]=(v2df){atmp[12][j],atmp[13][j]};
	    atk[7]=(v2df){atmp[14][j],atmp[15][j]};
	}
	//	END_TSC(t2,18);
	//	}			   int istart)
    }

}

void transpose_rowtocol16_3(int n, double a[][RDIM], double at[][n],
			 int istart)
{
    int i,j,k;
    const int m=16;
    double atmp[m][m];
    double atmp2[m][m];
    int mend;
    //    BEGIN_TSC;
    //#pragma omp parallel for private(i,j,k,atmp)
    for(i=istart;i<n;i+=m){
	for(k=0;k<m;k++){
	    v2df * ak = (v2df*) a[i+k];
	    v2df * akk = (v2df*) atmp[k];
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][0]):"memory");
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][8]):"memory");
	    //	    __builtin_prefetch(a[i+k+m*2],0,0);
	    //	    __builtin_prefetch(a[i+k+m*2]+8,0,0);
	    akk[0]  =ak[0];
	    akk[1]  =ak[1];
	    akk[2]  =ak[2];
	    akk[3]  =ak[3];
	    akk[4]  =ak[4];
	    akk[5]  =ak[5];
	    akk[6]  =ak[6];
	    akk[7]  =ak[7];
	}
	{
	for(j=0;j<m;j++){
	    v2df * atk= (v2df*)(atmp2[j]);
	    atk[0]=(v2df){atmp[0][j],atmp[1][j]};
	    atk[1]=(v2df){atmp[2][j],atmp[3][j]};
	    atk[2]=(v2df){atmp[4][j],atmp[5][j]};
	    atk[3]=(v2df){atmp[6][j],atmp[7][j]};
	    atk[4]=(v2df){atmp[8][j],atmp[9][j]};
	    atk[5]=(v2df){atmp[10][j],atmp[11][j]};
	    atk[6]=(v2df){atmp[12][j],atmp[13][j]};
	    atk[7]=(v2df){atmp[14][j],atmp[15][j]};
	}
	}
	{
	    
	for(j=0;j<m;j++){
	    v2df * atk= (v2df*)(at[j]+i);
	    v2df * attk= (v2df*)(atmp2[j]);
	    atk[0]=attk[0];
	    atk[1]=attk[1];
	    atk[2]=attk[2];
	    atk[3]=attk[3];
	    atk[4]=attk[4];
	    atk[5]=attk[5];
	    atk[6]=attk[6];
	    atk[7]=attk[7];
	}
	}
    }
    //    END_TSC(t,2);
}

void transpose_rowtocol16_4(int n, double a[][RDIM], double at[][n],
			 int istart)
{
    int i,j,k;
    const int m=16;
    const int mh=8;
    v2df atmp[m][mh];
    double atmp2[m][m];
    int mend;
    //    BEGIN_TSC;
    //#pragma omp parallel for private(i,j,k,atmp)
    for(i=istart;i<n;i+=m){
	for(k=0;k<m;k++){
	    v2df * ak = (v2df*) a[i+k];
	    v2df * akk =  atmp[k];
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][0]):"memory");
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][8]):"memory");
	    //	    __builtin_prefetch(a[i+k+m*2],0,0);
	    //	    __builtin_prefetch(a[i+k+m*2]+8,0,0);
	    akk[0]  =ak[0];
	    akk[1]  =ak[1];
	    akk[2]  =ak[2];
	    akk[3]  =ak[3];
	    akk[4]  =ak[4];
	    akk[5]  =ak[5];
	    akk[6]  =ak[6];
	    akk[7]  =ak[7];
	}
	{
	    for(j=0;j<m;j+=2){
		v2df * atk= (v2df*)(atmp2[j]);
		int jh = j>>1;
		
		//		atk[0]=__builtin_ia32_shufpd(atmp[0][jh],
		//		     atmp[1][jh],0x00);
		*(__m128d *)atk = _mm_shuffle_pd (*(__m128d *)(atmp[0]+jh),
						 *(__m128d *)(atmp[1]+jh),
						 0x00);
		atk[1]=__builtin_ia32_shufpd(atmp[2][jh],
					     atmp[3][jh],0x00);
		atk[2]=__builtin_ia32_shufpd(atmp[4][jh],
					     atmp[5][jh],0x00);
		atk[3]=__builtin_ia32_shufpd(atmp[6][jh],
					     atmp[7][jh],0x00);
		atk[4]=__builtin_ia32_shufpd(atmp[8][jh],
					     atmp[9][jh],0x00);
		atk[5]=__builtin_ia32_shufpd(atmp[10][jh],
					     atmp[11][jh],0x00);
		atk[6]=__builtin_ia32_shufpd(atmp[12][jh],
					     atmp[13][jh],0x00);
		atk[7]=__builtin_ia32_shufpd(atmp[14][jh],
					     atmp[15][jh],0x00);
		atk= (v2df*)(atmp2[j+1]);
		atk[0]=__builtin_ia32_shufpd(atmp[0][jh],
					     atmp[1][jh],0xff);
		atk[1]=__builtin_ia32_shufpd(atmp[2][jh],
					     atmp[3][jh],0xff);
		atk[2]=__builtin_ia32_shufpd(atmp[4][jh],
					     atmp[5][jh],0xff);
		atk[3]=__builtin_ia32_shufpd(atmp[6][jh],
					     atmp[7][jh],0xff);
		atk[4]=__builtin_ia32_shufpd(atmp[8][jh],
					     atmp[9][jh],0xff);
		atk[5]=__builtin_ia32_shufpd(atmp[10][jh],
					     atmp[11][jh],0xff);
		atk[6]=__builtin_ia32_shufpd(atmp[12][jh],
					     atmp[13][jh],0xff);
		atk[7]=__builtin_ia32_shufpd(atmp[14][jh],
					     atmp[15][jh],0xff);
	    }
	}
	{
	    
	    for(j=0;j<m;j++){
		v2df * atk= (v2df*)(at[j]+i);
		v2df * attk= (v2df*)(atmp2[j]);
		atk[0]=attk[0];
		atk[1]=attk[1];
		atk[2]=attk[2];
		atk[3]=attk[3];
		atk[4]=attk[4];
		atk[5]=attk[5];
		atk[6]=attk[6];
		atk[7]=attk[7];
	    }
	}
    }
    //    END_TSC(t,2);
}


void transpose_rowtocol(int n, double a[][RDIM],int m, double at[][n],
			int istart)
{
    int i,j,k;
    double atmp[m][m];
    BEGIN_TSC;
    if (m == 8){
	transpose_rowtocol8(n,a,at,istart);
        END_TSC(t,2);
	return;
    }
    if (m == 16){
	transpose_rowtocol16(n,a,at,istart);
        END_TSC(t,2);
	return;
    }
    for(i=istart;i<n;i+=m){
	for(k=0;k<m;k++){
	    for(j=0;j<m;j++){
		atmp[j][k]  =a[i+k][j];
	    }
	}
	for(j=0;j<m;j++){
	    for(k=0;k<m;k++){
		at[j][i+k]=atmp[j][k];
	    }
	}
    }
    END_TSC(t,2);
}

void transpose_coltorow8(int n, double a[][RDIM], double at[][n],
			int istart)
{
    int i,j,k;
    const int m=8;
    double atmp[m][m];
#pragma omp parallel for private(i,j,k,atmp)	
    for(i=istart;i<n;i+=m){
	for(j=0;j<m;j++){
	    double * atj = at[j]+i;
	    //	    __builtin_prefetch(at[j]+i+m+m,0,0);
	    // inserting prefetch here causes speed down...
	    atmp[0][j]  =atj[0];
	    atmp[1][j]  =atj[1];
	    atmp[2][j]  =atj[2];
	    atmp[3][j]  =atj[3];
	    atmp[4][j]  =atj[4];
	    atmp[5][j]  =atj[5];
	    atmp[6][j]  =atj[6];
	    atmp[7][j]  =atj[7];
	}
	for(k=0;k<m;k++){
	    v2df * atp = (v2df*) atmp[k];
	    v2df * ap = (v2df*) a[i+k];
	    *(ap)=*(atp);
	    *(ap+1)=*(atp+1);
	    *(ap+2)=*(atp+2);
	    *(ap+3)=*(atp+3);
	}
    }
}
void transpose_coltorow16(int n, double a[][RDIM], double at[][n],
			int istart)
{
    int i,j,k;
    const int m=16;
#pragma omp parallel for private(i,j,k)	
    for(i=istart;i<n;i+=m){
	double atmp[m][m];
	for(k=0;k<m;k++){
	    v2df * ak = (v2df*) (at[k]+i);
	    v2df * akk = (v2df*) atmp[k];
	    //	    asm("prefetchnta %0"::"m"(at[k][i+m*3]):"memory");
	    //	    asm("prefetchnta %0"::"m"(at[k][i+m*3+8]):"memory");
	    asm("prefetcht2 %0"::"m"(at[k][i+m*3]):"memory");
	    asm("prefetcht2 %0"::"m"(at[k][i+m*3+8]):"memory");
	    akk[0]  =ak[0];
	    akk[1]  =ak[1];
	    akk[2]  =ak[2];
	    akk[3]  =ak[3];
	    akk[4]  =ak[4];
	    akk[5]  =ak[5];
	    akk[6]  =ak[6];
	    akk[7]  =ak[7];
	}
	for(j=0;j<m;j++){
	    v2df * atk= (v2df*)(a[i+j]);
	    atk[0]=(v2df){atmp[0][j],atmp[1][j]};
	    atk[1]=(v2df){atmp[2][j],atmp[3][j]};
	    atk[2]=(v2df){atmp[4][j],atmp[5][j]};
	    atk[3]=(v2df){atmp[6][j],atmp[7][j]};
	    atk[4]=(v2df){atmp[8][j],atmp[9][j]};
	    atk[5]=(v2df){atmp[10][j],atmp[11][j]};
	    atk[6]=(v2df){atmp[12][j],atmp[13][j]};
	    atk[7]=(v2df){atmp[14][j],atmp[15][j]};
	}
    }
}
void transpose_coltorow16_0(int n, double a[][RDIM], double at[][n],
			int istart)
{
    int i,j,k;
    const int m=16;
    double atmp[m][m];
#pragma omp parallel for private(i,j,k,atmp)	
    for(i=istart;i<n;i+=m){
	for(j=0;j<m;j++){
	    double * atj = at[j]+i;
	    //	    __builtin_prefetch(at[j]+i+m+m,0,0);
	    // inserting prefetch here causes speed down...
	    atmp[0][j]  =atj[0];
	    atmp[1][j]  =atj[1];
	    atmp[2][j]  =atj[2];
	    atmp[3][j]  =atj[3];
	    atmp[4][j]  =atj[4];
	    atmp[5][j]  =atj[5];
	    atmp[6][j]  =atj[6];
	    atmp[7][j]  =atj[7];
	    atmp[8][j]  =atj[8];
	    atmp[9][j]  =atj[9];
	    atmp[10][j]  =atj[10];
	    atmp[11][j]  =atj[11];
	    atmp[12][j]  =atj[12];
	    atmp[13][j]  =atj[13];
	    atmp[14][j]  =atj[14];
	    atmp[15][j]  =atj[15];
	}
	for(k=0;k<m;k++){
	    v2df * atp = (v2df*) atmp[k];
	    v2df * ap = (v2df*) a[i+k];
	    *(ap)=*(atp);
	    *(ap+1)=*(atp+1);
	    *(ap+2)=*(atp+2);
	    *(ap+3)=*(atp+3);
	    *(ap+4)=*(atp+4);
	    *(ap+5)=*(atp+5);
	    *(ap+6)=*(atp+6);
	    *(ap+7)=*(atp+7);
	}
    }
}
void transpose_coltorow(int n, double a[][RDIM],int m, double at[][n],
			int istart)
{
    int i,j,k;
    double atmp[m][m];
    BEGIN_TSC;
    if (m == 8){
	transpose_coltorow8(n,a,at,istart);
	END_TSC(t,3);
	return;
    }
    if (m == 16){
	transpose_coltorow16(n,a,at,istart);
	END_TSC(t,3);
	return;
    }
    for(i=istart;i<n;i+=m){
	for(j=0;j<m;j++){
	    double * atj = at[j]+i;
	    for(k=0;k<m;k+=4){
		atmp[k][j]  =atj[k];
		atmp[k+1][j]  =atj[k+1];
		atmp[k+2][j]  =atj[k+2];
		atmp[k+3][j]  =atj[k+3];
	    }
	}
	for(k=0;k<m;k++){
	    double * aik = a[i+k];
	    for(j=0;j<m;j+=4){
		aik[j] = atmp[k][j];
		aik[j+1] = atmp[k][j+1];
		aik[j+2] = atmp[k][j+2];
		aik[j+3] = atmp[k][j+3];
	    }
	}
    }
    END_TSC(t,3);
}

void column_decomposition_with_transpose(int n,
				    double a[n][RDIM],
				    int m,
				    double awork[][n],
				    int pv[],
				    int i)
{
    int k,j;
    transpose_rowtocol(n, (double(*)[])  (&a[0][i]),m, awork,i);
    //    fprintf(stderr,"call cm column recursive %d %d\n", i, m);
    cm_column_decomposition_recursive( n, awork,m,pv,i);
    //    fprintf(stderr,"return cm column recursive %d %d\n", i, m);
    transpose_coltorow(n, (double(*)[])  (&a[0][i]),m, awork,i);
}


void column_decomposition_recursive(int n,
				    double a[n][RDIM],
				    int m,
				    double awork[][n],
				    int pv[],
				    int i)
{
    int  j, k;
    int ip,ii;
    double ainv;
    //    fprintf(stderr,"column recursive %d %d\n", i, m);
    if (m <= 16){
	// perform non-recursive direct decomposition
	BEGIN_TSC;
	column_decomposition_with_transpose(n, a, m,awork, pv,i);
	END_TSC(t,20);
	
    }else{	
	// process the left half by recursion
	column_decomposition_recursive(n, a, m/2, awork, pv,i);
	// process the right half
	process_right_part(n,a,m/2,awork, pv,i,i+m);
	column_decomposition_recursive(n, a, m/2, awork, pv+m/2,i+m/2);
	// process the swap of rows for the left half
	for(ii=i+m/2;ii<i+m;ii++){
	    swaprows(n,a,pv[ii-i],ii,i,i+m/2);
	}
	// normalize rows
	for(ii=i+m/2;ii<i+m;ii++){
	    scalerow(n,a,1.0/a[ii][ii] ,ii,i,i+m/2);
	}
    }
}
    
void lumcolumn( int n, double a[n][RDIM], double b[], int m,
		double awork[][n],int pv[],
		int recursive)
{
    int i;
    nswap=0;
    for(i=0;i<n;i+=m){
	BEGIN_TSC;
	//	fprintf(stderr,"lumcolumn i=%d\n", i);
	if (recursive){
	    column_decomposition_recursive(n, a, m, awork, pv,i);
	}else{
	    column_decomposition(n, a, m, pv,i);
	}
	//	fprintf(stderr,"lumcolumn column end\n");
	
	process_right_part(n,a,m,awork, pv,i,n+1);
	//	fprintf(stderr,"lumcolumn right end\n");
	END_TSC(t,19);

    }
    backward_sub(n,a,b);
}

typedef struct parmstruct{
    int n;
    int seed;
    int nb;
    int boardid;
    int nboards;
    int usehugepage;
} PARMS, *PPARMS;


void usage()
{
    fprintf(stderr,"lu2 options:\n");
    fprintf(stderr,"  -h: This help\n");
    fprintf(stderr,"  -s: seed (default=1)\n");
    fprintf(stderr,"  -n: size of matrix (default=8192)\n");
    fprintf(stderr,"  -b: block size (default=2048)\n");
    fprintf(stderr,"  -B: board id (default=0)\n");
    fprintf(stderr,"  -N: number of boards (default=1)\n");
    fprintf(stderr,"  -g: usehugetlbfs (default=no)\n");
}


extern char *optarg;
extern int optind;

void print_parms(FILE* stream, PPARMS parms)
{
    fprintf(stream,"N=%d Seed=%d NB=%d usehuge=%d\n",
	    parms->n,parms->seed,parms->nb,  parms->usehugepage);
    fprintf(stream,"Board id=%d # boards=%d\n",
	    parms->boardid, parms->nboards);
}

void read_parms(int argc, char * argv[], PPARMS parms)
{
    int ch;
    static struct option longopts[] = {
	{ "help",      no_argument,      0,           'h' },
	{ "block_size",      optional_argument,      NULL,           'b' },
	{ "board_id",      optional_argument,      NULL,           'B' },
	{ "nboards",      optional_argument,      NULL,           'N' },
	{ "seed",      optional_argument,            NULL,           's' },
	{ "ndim_matrix",   required_argument,      NULL,           'n' },
	{ "usehugepage",  no_argument,   0,     'g' },
	{ NULL,         0,                      NULL,           0 }
    };
    parms->seed=1;
    parms->n=8192;
    parms->nb = 2048;
    parms->boardid = 0;
    parms->nboards=1;
    parms->usehugepage =  0;
    while((ch=getopt_long(argc,argv,"B:N:b:ghn:s:",longopts, NULL))!= -1){
	fprintf(stderr,"optchar = %c optarg=%s\n", ch,optarg);
	switch (ch) {
	    case 'b': parms->nb = atoi(optarg); break;
	    case 'B': parms->boardid = atoi(optarg); break;
	    case 'N': parms->nboards = atoi(optarg); break;
	    case 'g': parms->usehugepage = 1; break;
	    case 's': parms->seed = atoi(optarg); break;
	    case 'n': parms->n = atoi(optarg); break;
	    case 'h': usage(); exit(1);
	    case '?':usage(); exit(1);
		break;
	    default:break;
	}
    }
    argc -= optind;
    argv += optind;
    print_parms(stderr, parms);
    print_parms(stdout, parms);
}

int main(int argc, char * argv[])
{
    int n, seed, nb, boardid;
    PARMS parms;
    int i;
    fprintf(stderr,"main top omp_max_threads=%d procs=%d\n",
		omp_get_max_threads(),omp_get_num_procs());
    read_parms(argc, argv, &parms);
    n = parms.n;
    nb = parms.nb;
    seed = parms.seed;
    boardid = parms.boardid;
    gdrsetboardid(parms.boardid);
    gdrsetnboards(parms.nboards);
#if 0    
    fprintf(stderr, "Enter n, seed, nb:");
    scanf("%d%d%d", &n, &seed, &nb);
    printf("N=%d Seed=%d NB=%d\n", n,seed,nb);
#endif    
    double (*a)[];
    double (*acopy)[];
    double (*awork)[];
    int * pv;
    double *b, *bcopy;
    long int nl=n;
    if (parms.usehugepage){
	char fname[128];
	sprintf(fname,"/mnt/huge/aaa-%d",boardid);
	int fd = open(fname, O_RDWR|O_CREAT, 0777);
	size_t size = ((long)(sizeof(double)*((long)nl)*
			      (long)(RDIM))+0x400000)&0xffffffffffc00000L;
	a = (double(*)[]) mmap(0, size, PROT_READ|PROT_WRITE,
			       MAP_SHARED, fd, 0);
	size_t worksize = ((sizeof(double)*nb*n)+0x400000)&0xffc00000;
	off_t offset = (off_t) size;
	awork = (double(*)[]) mmap(0, worksize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
	//    printf("a, awork offset size= %lx %lx %lx %lx %lx\n",
	//	   (long) (a), (long) (awork),
	//	   (long) (awork)-(long) (a), (long) offset, (long)(size));
	//    printf("size of size_t and off_t long= %d %d %d\n", sizeof(size_t),
	//	   sizeof(off_t), sizeof(long));
    }else{
	a = (double(*)[]) malloc(sizeof(double)*n*(RDIM));
	awork = (double(*)[]) malloc(sizeof(double)*nb*n);
    }
    b = (double*)malloc(sizeof(double)*n);
    bcopy = (double*)malloc(sizeof(double)*n);
    pv = (int*)malloc(sizeof(int)*n);
    reset_gdr(RDIM, a, nb, awork, n);
    
    if (seed == 0){
	readmat(n,a);
    }else{
	randomsetmat(n,seed,a);
    }
    fprintf(stderr,"read/set mat end\n");
    //    copymats(n,a,acopy);
    //    copybvect(n,a,bcopy);
    fprintf(stderr,"copy mat end\n");
    //    printmat(n,a,b);
    //    lu2columnv2(n,a,b);
    //    lu2columnv2(n,a,b);
    //    lub(n,a,b,NBK);
    //    printmat(n,a,b);
    //    showresult(n,acopy, b, bcopy);
    //    copymats(n,acopy,bcopy,a, b);
    //    lu(n,a,b);
    timer_init();
    init_timer();

    fprintf(stderr,"before lumcolumn omp_max_threads=%d procs=%d\n",
		omp_get_max_threads(),omp_get_num_procs());

    lumcolumn(n,a,b,nb,awork,pv,1);
    //    lu(n,a,b);
    double ctime=cpusec();
    double wtime=wsec();
    if (seed == 0){
	readmat(n,a);
    }else{
	randomsetmat(n,seed,a);
    }
    showresult(n,a, b);
    double nd = n;
    double speed = nd*nd*nd*2.0/3.0/wtime/1e9;
    printf("Nswap=%d cpsec =  %g wsec=%g %g Gflops\n", nswap,  ctime, wtime,
	   speed);
    print_timers((double)n, (double)nb );
    return 0;
}
