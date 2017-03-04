//#define DO_RCOMM_LAST	
//#define CONCURRENT_UCOMM		
// lu2_mpi.c
//
// test program for blocked LU decomposition
//
// Time-stamp: <11/06/25 12:58:48 makino>
//#define NOBLAS
//#define TIMETEST
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include "lu2_mp.h"

#include <emmintrin.h>
typedef double v2df __attribute__((vector_size(16)));
typedef union {v2df v; double s[2];}v2u;

#include <lu2tlib.h>
#include <lu2tlib.h>
#include <lu2lib.h>
void timer_init();
double cpusec();
double wsec();

omp_lock_t my_lock;

void printmat_MP(int nnrow, int nncol, double a[nnrow][nncol], 
		 PCONTROLS controls,  PPARMS parms);

#define RDIM (n+16)
void copymats( int n, double a[n][RDIM], double a2[n][RDIM])
{
    int i, j;
    for(i=0;i<n;i++){
	for(j=0;j<n+2;j++) a2[i][j] = a[i][j];
    }
}

void dumpsubmat(char * s,int n1, double mat[][n1], int nrow, int ncolumn, PPARMS parms, PCONTROLS controls)
{
    int ip, i, j;
    return;
    fprintf(stderr, "npr, npc = %d %d\n", parms->nprow, parms->npcol);
    sleep(MP_myprocid()*2+1);
    fprintf(stderr,"\n\n\n\n\n");
    fprintf(stderr,"%s\n", s);
    for(i=0;i<nrow;i++){
	fprintf(stderr,"%3d ", i);
	for(j=0;j<ncolumn;j++) fprintf(stderr," %6.2f", mat[i][j]);
	fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n\n\n\n\n");
    fflush(stderr);
}
void copysubmat0(int n1, double src[][n1], int n2, double dest[][n2],
		 int nrow, int ncolumn )
{
    int i, j;
    for(i=0;i<nrow;i++){
	for(j=0;j<ncolumn;j++) dest[i][j] = src[i][j];
    }
}

void copysubmat8(int n1, double src[][n1], int n2, double dest[][n2],
		int nrow, int ncolumn )
{
    int i;
    if (nrow * ncolumn > 10000){
#pragma omp parallel for private(i) schedule(static,64)
	for(i=0;i<nrow;i++){
	    v2df * s = (v2df*) (src[i]);
	    v2df * d = (v2df*) (dest[i]);
	    __builtin_prefetch(src[i+8],0,0);
	    d[0]=s[0];
	    d[1]=s[1];
	    d[2]=s[2];
	    d[3]=s[3];
	}
    }else{
	for(i=0;i<nrow;i++){
	    v2df * s = (v2df*) (src[i]);
	    v2df * d = (v2df*) (dest[i]);
	    __builtin_prefetch(src[i+8],0,0);
	    d[0]=s[0];
	    d[1]=s[1];
	    d[2]=s[2];
	    d[3]=s[3];
	}
    }
}
void copysubmat16(int n1, double src[][n1], int n2, double dest[][n2],
		int nrow, int ncolumn )
{
    int i;
    if (nrow * ncolumn > 10000){
#pragma omp parallel for private(i)  schedule(static,64)
	for(i=0;i<nrow;i++){
	    v2df * s = (v2df*) (src[i]);
	    v2df * d = (v2df*) (dest[i]);
	    __builtin_prefetch(src[i+8],0,0);
	    d[0]=s[0];
	    d[1]=s[1];
	    d[2]=s[2];
	    d[3]=s[3];
	    d[4]=s[4];
	    d[5]=s[5];
	    d[6]=s[6];
	    d[7]=s[7];
	}
    }else{
	for(i=0;i<nrow;i++){
	    v2df * s = (v2df*) (src[i]);
	    v2df * d = (v2df*) (dest[i]);
	    __builtin_prefetch(src[i+8],0,0);
	    d[0]=s[0];
	    d[1]=s[1];
	    d[2]=s[2];
	    d[3]=s[3];
	    d[4]=s[4];
	    d[5]=s[5];
	    d[6]=s[6];
	    d[7]=s[7];
	}
    }
}

void copysubmat(int n1, double src[][n1], int n2, double dest[][n2],
		int nrow, int ncolumn )
{
    // assume that ncolum is multiple of 8 and
    // address is 16-byte aligined

    int i;
    int j;

    if (ncolumn < 8){
	copysubmat0( n1, src, n2, dest, nrow, ncolumn);
	return;
    }

#if 1
    if (ncolumn == 8){
	copysubmat8( n1, src, n2, dest, nrow, ncolumn);
	return;
    }
    if (ncolumn == 16){
	copysubmat16( n1, src, n2, dest, nrow, ncolumn);
	return;
    }
#endif
    BEGIN_TIMER(t);
    if (nrow * ncolumn > 30000){
#pragma omp parallel for private(i)  schedule(static,64)
	for(i=0;i<nrow;i++){
	    int j;
	    //	for(j=0;j<ncolumn;j++) dest[i][j] = src[i][j];
	    v2df * s = (v2df*) (src[i]);
	    v2df * d = (v2df*) (dest[i]);
	    for(j=0;j<ncolumn/2;j+=8){
		//		__builtin_prefetch(s+j+96,0,0);
#if 0
		d[j]=s[j];
		d[j+1]=s[j+1];
		d[j+2]=s[j+2];
		d[j+3]=s[j+3];
		d[j+4]=s[j+4];
		d[j+5]=s[j+5];
		d[j+6]=s[j+6];
		d[j+7]=s[j+7];
#else
		__builtin_ia32_movntpd((double*)&d[j], s[j]);
		__builtin_ia32_movntpd((double*)&d[j+1], s[j+1]);
		__builtin_ia32_movntpd((double*)&d[j+2], s[j+2]);
		__builtin_ia32_movntpd((double*)&d[j+3], s[j+3]);
		__builtin_ia32_movntpd((double*)&d[j+4], s[j+4]);
		__builtin_ia32_movntpd((double*)&d[j+5], s[j+5]);
		__builtin_ia32_movntpd((double*)&d[j+6], s[j+6]);
		__builtin_ia32_movntpd((double*)&d[j+7], s[j+7]);
#endif	    
	    }
	}
    }else{
	for(i=0;i<nrow;i++){
	    int j;
	    //	for(j=0;j<ncolumn;j++) dest[i][j] = src[i][j];
	    v2df * s = (v2df*) (src[i]);
	    v2df * d = (v2df*) (dest[i]);
	    for(j=0;j<ncolumn/2;j+=8){
#if 0		
		d[j]=s[j];
		d[j+1]=s[j+1];
		d[j+2]=s[j+2];
		d[j+3]=s[j+3];
		d[j+4]=s[j+4];
		d[j+5]=s[j+5];
		d[j+6]=s[j+6];
		d[j+7]=s[j+7];
#else		
		__builtin_ia32_movntpd((double*)&d[j], s[j]);
		__builtin_ia32_movntpd((double*)&d[j+1], s[j+1]);
		__builtin_ia32_movntpd((double*)&d[j+2], s[j+2]);
		__builtin_ia32_movntpd((double*)&d[j+3], s[j+3]);
		__builtin_ia32_movntpd((double*)&d[j+4], s[j+4]);
		__builtin_ia32_movntpd((double*)&d[j+5], s[j+5]);
		__builtin_ia32_movntpd((double*)&d[j+6], s[j+6]);
		__builtin_ia32_movntpd((double*)&d[j+7], s[j+7]);
#endif
	    }
	}	
    }
    END_TIMER(t,43,nrow*ncolumn*1.0);
#if 0
    for(i=0; i< nrow; i++){
	for(j=0; j<ncolumn; j++){
	    printf(" %6.2f", src[i][j]);
	}
	printf("\n");
    }
    printf("\n\n");
    for(i=0; i< nrow; i++){
	for(j=0; j<ncolumn; j++){
	    printf(" %6.2f", dest[i][j]);
	}
	printf("\n");
    }
    printf("\n");
#endif    

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
	for(j=0;j<n;j++) {
	    double * ap = a[i];
	    ap[j]=drand48()-0.5;
	}
	//	printf("n, i=%d\n", i);
	a[i][n]=drand48()-0.5;;
    }
}

void MP_randomsetmat(int nncol, int nnrow,double a[nnrow][nncol],
		     PPARMS parms, PCONTROLS controls,
		     int set_b, double b[])
{
    long int i, j;
    int seed;
    MP_message("enter MP_randomsetmat");
    srandom(parms->seed*(MP_myprocid()+1));
    for(i=0; i< MP_myprocid()+100; i++) seed = random();
    srand48((long) seed);
    dprintf(9,"seed, nncol,nnrow, ncols, nrows = %d %d %d %d %d\n",
	    seed,nncol, nnrow, controls->ncol, controls->nrow);
    for(i=0;i<controls->nrow;i++){
	//	printf("i=%d\n", i);
	for(j=0;j<controls->ncol;j++) {
	    double * ap = a[i];
	    ap[j]=drand48()-0.5;
	    //	    dprintf(9,"ap %d %d = %g\n", i,j, ap[j]);
	}
	//	printf("n, i=%d\n", i);
    }
    for(i=0;i<controls->nrow;i++){
	//	b[i]=1;
	b[i]=drand48()-0.5;
    }
    MPI_Bcast(b, controls->nrow, MPI_DOUBLE, 0, controls->row_comm);

    if (set_b){
	for(i=0;i<controls->nrow;i++){
	    a[i][controls->ncol]=b[i];
	}
    }
    
    MP_message("end MP_randomsetmat");
}

void calclocalnorm(int nncol, int nnrow,double a[nnrow][nncol],
		   int nrow, int ncol, 
		   double ao[], double a1[])
{
    int i, j;
    //    fprintf(stderr, "nncol=%d nnrow=%d ncol=%d nrow=%d\n", nncol, nnrow, ncol, nrow);
    for(j=0;j<nrow;j++) ao[j]=0;
    for(i=0;i<ncol;i++) a1[i]=0;
    for(j=0;j<nrow; j++){
	for(i=0;i<ncol;i++) { 
	    double aa = fabs(a[j][i]);
	    ao[j] += aa;
	    a1[i] += aa;
	}
    }
}


double vnormi(double a[], int n)
{
    double x = 0;
    int i;
    for(i=0;i<n;i++){
	double y = fabs(a[i]);
	if(y > x) x = y;
    }
    return x;
}


void MP_calcnorm(int nncol, int nnrow,double a[nnrow][nncol],
		 PPARMS parms, PCONTROLS controls,
		 double *norm1, double *norminf)
{
    int i, j;
    int nrow = controls->nrow;
    int ncol = controls->ncol;
    MP_message("enter MP_calcnorm");
    double ao[nnrow];
    double aosum[nnrow];
    double a1[nncol];
    double a1sum[nncol];
    print_current_time("call calculocalnorm");
    calclocalnorm(nncol, nnrow, a, nrow, ncol, ao, a1);
    print_current_time("end  calculocalnorm");
    MPI_Allreduce(ao,aosum, nrow, MPI_DOUBLE, MPI_SUM, controls->row_comm);
    MPI_Allreduce(a1,a1sum, ncol, MPI_DOUBLE, MPI_SUM, controls->col_comm);
    double aolocal = vnormi(aosum, nrow);
    double a1local = vnormi(a1sum, ncol);
    double aoglobal, a1global;
    MPI_Allreduce(&aolocal, &aoglobal, 1, MPI_DOUBLE, MPI_MAX,
		  controls->col_comm);
    MPI_Allreduce(&a1local, &a1global, 1, MPI_DOUBLE, MPI_MAX, 
		  controls->col_comm);
    *norm1 = a1global;
    *norminf = aoglobal;
    MP_message("end MP_calcnorm");
}

void MP_calcvnorm(double b[], PPARMS parms, PCONTROLS controls,
		  double *norm1, double *norminf)
{
    int i, j;
    int nrow = controls->nrow;
    MP_message("enter MP_calcvnorm");
    double bo=0;
    double b1=0;
    for(i=0;i<nrow; i++)b1 += fabs(b[i]);
    bo = vnormi(b, nrow);
    double b1sum, boglobal;
    MPI_Allreduce(&b1,&b1sum, 1, MPI_DOUBLE, MPI_SUM, controls->col_comm);
    MPI_Allreduce(&bo, &boglobal, 1, MPI_DOUBLE, MPI_MAX, controls->col_comm);
    *norm1 = b1sum;
    *norminf = boglobal;
    MP_message("end MP_calcvnorm");
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

int MP_find_pivot(int nnrow, int nncol, double a[nnrow][nncol],
		  PPARMS parms,
		  PCONTROLS controls,
		  int current)
{
    int index;
    //    fprintf(stderr, "%d: hav_current_col = %d\n", MP_myprocid(),
    //	    have_current_col(current,  parms, controls));
    
    if (have_current_col(current,  parms, controls)){
	int first=first_row(current, parms, controls);
	double fmax =0;
	int imax = -1;
	int i;
	int ii = local_colid(current, parms, controls);
	for(i=first; i< controls->nrow;i++){
	    if (fabs(a[i][ii]) > fmax){
		fmax = fabs(a[i][ii]);
		imax = i;
	    }
	}
	index = global_rowid(imax, parms,  controls);// <- wrong!!
	MP_max_and_maxloc(&fmax, &index, controls->col_comm);
    }
    
    MPI_Bcast(&index,sizeof(int),MPI_BYTE,
	      pcolid(current,parms,controls),
	      controls->row_comm);
    MPI_Barrier(MPI_COMM_WORLD);
    //    fprintf(stderr,"MP find pivot returns %d\n", index);
    return index;
}
int MP_find_pivot_transposed(int nnrow, int nncol, double a[nncol][nnrow],
		  PPARMS parms,
		  PCONTROLS controls,
		  int current)
{
    int index;
    //    fprintf(stderr, "%d: hav_current_col = %d\n", MP_myprocid(),
    //	    have_current_col(current,  parms, controls));
    if (have_current_col(current,  parms, controls)){
	int first=first_row(current, parms, controls);
	double fmax =0;
	int imax = -1;
	int i;
	int ii = local_colid(current, parms, controls)%nncol;
#if 0	
	for(i=first; i< controls->nrow;i++){
	    if (fabs(a[ii][i]) > fmax){
		fmax = fabs(a[ii][i]);
		imax = i;
	    }
	}
#else
	if (controls->nrow-first > 0){
	    imax = cblas_idamax(controls->nrow-first, a[ii]+first, 1)+first;
	    fmax = fabs(a[ii][imax]);
	}else{
	    imax = 0;
	    fmax = -1;
	}
#endif	
	index = global_rowid(imax, parms,  controls);// <- wrong!!
	//	fprintf(stderr,"find_pivot local max, index =%10.3e, %d\n", fmax, index);
	MP_max_and_maxloc(&fmax, &index, controls->col_comm);
	//	fprintf(stderr,"find_pivot g max, index =%10.3e, %d\n", fmax, index);

    }
    return index;
}

void local_row_exchange(int nnrow,int nncol,double a[nnrow][nncol],
			int row1,int row2)
{
    int i;
    double *ap1 = a[row1];
    double *ap2 = a[row2];
    for(i=0;i<nncol;i++){
	double tmp=ap1[i];
	ap1[i]=ap2[i];
	ap2[i]=tmp;
    }
}

void local_row_exchange_blocked(int nnrow,int nncol,double a[nnrow][nncol],
				int row1,int row2, int c1, int c2)
{
    int i;
    double *ap1 = a[row1];
    double *ap2 = a[row2];
    for(i=c1;i<c2;i++){
	double tmp=ap1[i];
	ap1[i]=ap2[i];
	ap2[i]=tmp;
    }
}

void local_row_exchange_blocked_transposed(int nnrow,int nncol,
					   double a[nncol][nnrow],
					   int row1,int row2,
					   int c1, int c2)
{
    int i;
    double *ap1 = a[row1];
    double *ap2 = a[row2];
    for(i=c1;i<c2;i++){
	double tmp=a[i][row1];
	a[i][row1]=a[i][row2];
	a[i][row2]=tmp;
    }
}

	

void MP_swap_row_ptop(int nnrow, int nncol, double a[nnrow][nncol],
		      int myrow, int procswap,
		      PCONTROLS controls)
{
    double atmp[nncol];
    int i;
    MPI_Status mpstatus;
    for(i=0;i<nncol;i++)atmp[i]=a[myrow][i];
    MPI_Sendrecv(atmp, nncol, MPI_DOUBLE, procswap, MPSWAPTAG,
		 a[myrow],nncol, MPI_DOUBLE, procswap, MPSWAPTAG,
		 controls->col_comm, &mpstatus);
}


void MP_swap_row_ptop_blocked(int nnrow, int nncol, double a[nnrow][nncol],
			      int myrow, int procswap,  PCONTROLS controls,
			      int c1, int c2)
{

    double atmp[nncol];
    int i;
    MPI_Status mpstatus;
    if (c2 > c1){
	int ndata = c2-c1;
	for(i=0;i<ndata;i++)atmp[i]=a[myrow][i+c1];
	MPI_Sendrecv(atmp, ndata, MPI_DOUBLE, procswap, MPSWAPTAG,
		     a[myrow]+c1,ndata, MPI_DOUBLE, procswap, MPSWAPTAG,
		     controls->col_comm, &mpstatus);
    }
}

void MP_swap_row_ptop_blocked_transposed(int nnrow, int nncol,
					 double a[nncol][nnrow],
					 int myrow, int procswap,
					 PCONTROLS controls,
					 int c1, int c2)
{

    double atmp[nncol];
    double atmp2[nncol];
    int i;
    MPI_Status mpstatus;
    if (c2 > c1){
	int ndata = c2-c1;
	for(i=0;i<ndata;i++)atmp[i]=a[i+c1][myrow];
	MPI_Sendrecv(atmp, ndata, MPI_DOUBLE, procswap, MPSWAPTAG,
		     atmp2,ndata, MPI_DOUBLE, procswap, MPSWAPTAG,
		     controls->col_comm, &mpstatus);
	for(i=0;i<ndata;i++)a[i+c1][myrow]=atmp2[i];
    }
}



int MP_swap_rows(int nnrow, int nncol, double a[nnrow][nncol],
		  PPARMS parms,
		  PCONTROLS controls,
		 int current,
		 int pivot)
{
    int current_proc;
    int current_lloc;
    int pivot_proc;
    int pivot_lloc;
    convert_global_index_to_local_rows(current, &current_proc, &current_lloc,
				       parms, controls);
    convert_global_index_to_local_rows(pivot, &pivot_proc, &pivot_lloc,
				       parms, controls);

    if (current_proc == pivot_proc){
	if (current_proc == controls->rank_in_col)
	    local_row_exchange(nnrow,nncol,a,current_lloc,pivot_lloc);
    }else if (current_proc == controls->rank_in_col){
	MP_swap_row_ptop(nnrow,nncol,a,current_lloc,pivot_proc, controls);
    }else if (pivot_proc == controls->rank_in_col){
	MP_swap_row_ptop(nnrow,nncol,a,pivot_lloc,current_proc, controls);
    }
}

int MP_swap_rows_blocked(int nnrow, int nncol, double a[nnrow][nncol],
			 PPARMS parms,  PCONTROLS controls,
			 int current, int pivot,
			 int c1, int c2)
{
    int current_proc;
    int current_lloc;
    int pivot_proc;
    int pivot_lloc;
    convert_global_index_to_local_rows(current, &current_proc, &current_lloc,
				       parms, controls);
    convert_global_index_to_local_rows(pivot, &pivot_proc, &pivot_lloc,
				       parms, controls);
    if (current_proc == pivot_proc){
	if (current_proc == controls->rank_in_col)
	    local_row_exchange_blocked(nnrow,nncol,a,current_lloc,
				       pivot_lloc,c1,c2);
    }else if (current_proc == controls->rank_in_col){
	MP_swap_row_ptop_blocked(nnrow,nncol,a,current_lloc,pivot_proc,
				 controls,c1,c2);
    }else if (pivot_proc == controls->rank_in_col){
	MP_swap_row_ptop_blocked(nnrow,nncol,a,pivot_lloc,current_proc,
				 controls,c1,c2);
    }
}

int MP_swap_rows_blocked_transposed(int nnrow, int nncol,
				    double a[nncol][nnrow],
			 PPARMS parms,  PCONTROLS controls,
			 int current, int pivot,
			 int c1, int c2)
{
    int current_proc;
    int current_lloc;
    int pivot_proc;
    int pivot_lloc;
    convert_global_index_to_local_rows(current, &current_proc, &current_lloc,
				       parms, controls);
    convert_global_index_to_local_rows(pivot, &pivot_proc, &pivot_lloc,
				       parms, controls);
    //    dprintf(9,"swap_rows_transposed cp,pp,rank_in_col=%d %d %d\n",
    //	    current_proc, pivot_proc,controls->rank_in_col);
    if (current_proc == pivot_proc){
	if (current_proc == controls->rank_in_col)
	    local_row_exchange_blocked_transposed(nnrow,nncol,a,current_lloc,
				       pivot_lloc,c1,c2);
    }else if (current_proc == controls->rank_in_col){
	MP_swap_row_ptop_blocked_transposed(nnrow,nncol,a,
					    current_lloc,pivot_proc,
					    controls,c1,c2);
    }else if (pivot_proc == controls->rank_in_col){
	MP_swap_row_ptop_blocked_transposed(nnrow,nncol,a,pivot_lloc,
					    current_proc,
					    controls,c1,c2);
    }
//    dprintf(9,"swap_rows_transposed end\n");
}

int MP_scale_row(int nnrow, int nncol, double a[nnrow][nncol],
		  PPARMS parms,
		  PCONTROLS controls,
		 int current)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	double scale = a[current_lrow][current_lcol];
	int i;
	MPI_Bcast(&scale,sizeof(double),MPI_BYTE,
		  current_pcol,
		  controls->row_comm);
	scale = 1.0/scale;
	for(i=0;i<nncol;i++)a[current_lrow][i] *= scale;
    }
}

int MP_scale_row_blocked(int nnrow, int nncol, double a[nnrow][nncol],
			 PPARMS parms, PCONTROLS controls, int current,
			 int c1, int c2, int singlecol)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	double scale = a[current_lrow][current_lcol];
	int i;
	if (! singlecol){
	    MPI_Bcast(&scale,sizeof(double),MPI_BYTE,
		      current_pcol,
		      controls->row_comm);
	}
	scale = 1.0/scale;
	if (controls->rank_in_row != current_pcol){
	    for(i=c1;i<c2;i++)a[current_lrow][i] *= scale;
	}else{
	    for(i=c1;i<c2;i++)
		if (i != current_lcol)
		    a[current_lrow][i] *= scale;
	}
    }
}
int MP_scale_row_blocked_using_scale(int nnrow, int nncol, double a[nnrow][nncol],
			 PPARMS parms, PCONTROLS controls, int current,
				     int c1, int c2, int singlecol,double scaleval)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	int i;
	for(i=c1;i<c2;i++)a[current_lrow][i] *= scaleval;
    }
}


double scaleval(int nnrow, int nncol, double a[nnrow][nncol],
		PPARMS parms, PCONTROLS controls, int current)
{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    double scale = a[current_lrow][current_lcol];
    return 1.0/scale;
}

int MP_construct_scalevector(int nnrow, int nncol, double a[nnrow][nncol],
			     PPARMS parms, PCONTROLS controls, int current,
			     int nb,
			     double scale[])
{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	int i;
	for(i=0;i<nb;i++){
	    scale[i]= 1.0/ a[current_lrow+i][current_lcol+i];
	}
    }
}

int MP_scale_row_blocked_transposed(int nnrow, int nncol,
				    double a[nncol][nnrow],
			 PPARMS parms, PCONTROLS controls, int current,
			 int c1, int c2)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    current_lcol %= nncol;
    if (current_prow == controls->rank_in_col){
	double scale = a[current_lcol][current_lrow];
	int i;
	scale = 1.0/scale;
	if (controls->rank_in_row != current_pcol){
	    for(i=c1;i<c2;i++)a[i][current_lrow] *= scale;
	}else{
	    for(i=c1;i<c2;i++)
		if (i != current_lcol)
		    a[i][current_lrow] *= scale;
	}
    }
}


int MP_update_single(int nnrow, int nncol, double a[nnrow][nncol],
		     PPARMS parms,
		     PCONTROLS controls,
		     int current)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j;
    double arow[nncol];
    double acol[nnrow];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startcol;
    int startrow;
    int nb = parms->nb;
    if (current_prow > controls->rank_in_col){
	startrow = (current_lrow/nb +1)*nb;
    }else if (current_prow == controls->rank_in_col){
	startrow =current_lrow+1;
    }else{
	startrow = (current_lrow/nb)*nb;
    }
    if (current_pcol > controls->rank_in_row){
	startcol = (current_lcol/nb +1)*nb;
    }else if (current_pcol == controls->rank_in_row){
	startcol =current_lcol+1;
    }else{
	startcol = (current_lcol/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	for (i=0;i<nrow-startrow;i++)acol[i]=a[i+startrow][current_lcol];
    }
    if (current_prow == controls->rank_in_col){
	for (i=0;i<ncol-startcol;i++)arow[i]=a[current_lrow][i+startcol];
    }
    MPI_Bcast(arow,sizeof(double)*(ncol-startcol),MPI_BYTE,
	      current_prow,  controls->col_comm);
    MPI_Bcast(acol,sizeof(double)*(nrow-startrow),MPI_BYTE,
	      current_pcol,  controls->row_comm);
    for (i=startcol;i<ncol;i++){
	for (j=startrow;j<nrow;j++){
	    a[j][i] -= acol[j-startrow]*arow[i-startcol];
	}
    }
}


int MP_update_single_blocked(int nnrow, int nncol, double a[nnrow][nncol],
			     PPARMS parms,  PCONTROLS controls,
			     int current, int c1, int c2)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j;
    double arow[nncol];
    double acol[nnrow];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (current_prow > controls->rank_in_col){
	startrow = (current_lrow/nb +1)*nb;
    }else if (current_prow == controls->rank_in_col){
	startrow =current_lrow+1;
    }else{
	startrow = (current_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	for (i=0;i<nrow-startrow;i++)acol[i]=a[i+startrow][current_lcol];
    }
    if (current_prow == controls->rank_in_col){
	for (i=0;i<c2-c1;i++)arow[i]=a[current_lrow][i+c1];
    }
    MPI_Bcast(arow,sizeof(double)*(c2-c1),MPI_BYTE,
	      current_prow,  controls->col_comm);
    MPI_Bcast(acol,sizeof(double)*(nrow-startrow),MPI_BYTE,
	      current_pcol,  controls->row_comm);
    for (i=c1;i<c2;i++){
	for (j=startrow;j<nrow;j++){
	    a[j][i] -= acol[j-startrow]*arow[i-c1];
	}
    }
}

static void vsmulandsub0(int r0, int r1, double al[], double ar[], double s)
{
    int i;
    for (i=r0;i<r1;i++)al[i] -= ar[i]*s;
}

static void vsmulandsub(int r0, int r1, double al[], double ar[], double s)
{
    int j;
    if (r1 - r0 < 16){
	vsmulandsub0(r0, r1, al, ar, s);
	return;
    }
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

int MP_update_single_blocked_transposed(int nnrow, int nncol,
					double a[nncol][nnrow],
					PPARMS parms,  PCONTROLS controls,
					int current, int c1, int c2,
					double * arow, double * acol)
    
{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    //    dprintf(9,"enter MP_update_single_blocked_transposed\n");
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    current_lcol %= nncol;
    int startrow;
    int nb = parms->nb;
    if (current_prow > controls->rank_in_col){
	startrow = (current_lrow/nb +1)*nb;
    }else if (current_prow == controls->rank_in_col){
	startrow =current_lrow+1;
    }else{
	startrow = (current_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	for (i=0;i<nrow-startrow;i++)acol[i+startrow]=a[current_lcol][i+startrow];
    }
    //    if (current_prow == controls->rank_in_col){
    for (i=0;i<c2-c1;i++)arow[i]=a[i+c1][current_lrow];
    //    }
    // Since the value of current_prow is broadcasted, there
    // is no need for other processors to execute the above
    // loop. However, OpenMPI complains....
    
    
    //    dprintf(9," MP_update_single_blocked_transposed bcast %d\n",c2-c1);
    MPI_Bcast(arow,sizeof(double)*(c2-c1),MPI_BYTE,
	      current_prow,  controls->col_comm);
    //    dprintf(9," MP_update_single_blocked_transposed end bcast %d\n",c2-c1);
    // the following way does not look optimal yet...
    int ii;
#pragma omp parallel for private(ii) schedule(static)
    for(ii=0;ii<4;ii++){
	int k;
	int nr = (nrow-startrow)/4;
	int i1 = nr*ii+startrow;
	int i2 = i1+nr;
	if (ii==3) i2 =nrow;
	for (k=c1;k<c2;k++){
	    vsmulandsub(i1, i2, a[k],acol,arow[k-c1]);
	}
    }	
    //    dprintf(9,"end   MP_update_single_blocked_transposed\n");
}

int MP_update_single_blocked_global(int nnrow, int nncol,
				    double a[nnrow][nncol],
				    PPARMS parms,  PCONTROLS controls,
				    int current, int c1, int c2,
				    int global_startrow)

{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j;
    double arow[nncol];
    double acol[nnrow];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	for (i=0;i<nrow-startrow;i++)acol[i]=a[i+startrow][current_lcol];
    }
    if (current_prow == controls->rank_in_col){
	for (i=0;i<c2-c1;i++)arow[i]=a[current_lrow][i+c1];
    }
    MPI_Bcast(arow,sizeof(double)*(c2-c1),MPI_BYTE,
	      current_prow,  controls->col_comm);
    MPI_Bcast(acol,sizeof(double)*(nrow-startrow),MPI_BYTE,
	      current_pcol,  controls->row_comm);
    for (i=c1;i<c2;i++){
	for (j=startrow;j<nrow;j++){
	    a[j][i] -= acol[j-startrow]*arow[i-c1];
	}
    }
}
int MP_process_lmat(int nnrow, int nncol, double a[nnrow][nncol],
		    PPARMS parms,  PCONTROLS controls, int current, int nbin,
		    int global_startrow,  double acol[][nbin])
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	copysubmat(nncol, (double(*)[])(a[startrow]+current_lcol), nbin, acol,
		   nrow-startrow, nbin);
    }
    MPI_Bcast(acol,sizeof(double)*(nrow-startrow)*nbin,MPI_BYTE,
	      current_pcol,  controls->row_comm);
    return nrow-startrow;
}

int MP_prepare_lmat(int nnrow, int nncol, double a[nnrow][nncol],
		    PPARMS parms,  PCONTROLS controls, int current, int nbin,
		    int global_startrow,  double acol[][nbin])
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	for (i=0;i<nrow-startrow;i++)
	    for(j=0;j<nbin;j++)
		acol[i][j]=a[i+startrow][current_lcol+j];
    }
    return nrow-startrow;
}

    
int MP_update_multiple_blocked_global_using_lmat(int nnrow, int nncol,
						 double a[nnrow][nncol],
						 PPARMS parms,
						 PCONTROLS controls,
						 int current, int nbin,
						 int c1, int c2,
						 int global_startrow,
						 double acol[][nbin])
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    double arow[nbin][c2-c1];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    
    if (current_prow == controls->rank_in_col){
	for (i=0;i<c2-c1;i++)
	    for(j=0;j<nbin;j++)
		arow[j][i]=a[current_lrow+j][i+c1];
    }
    MPI_Bcast(arow,sizeof(double)*(c2-c1)*nbin,MPI_BYTE,
	      current_prow,  controls->col_comm);

    mydgemm(nrow-startrow, c2-c1, nbin, -1.0, &(acol[0][0]), nbin,
	    &(arow[0][0]), c2-c1, 1.0, &(a[startrow][c1]), nncol );

}


int MP_bcast_umat(int nnrow, int nncol,
		  double a[nnrow][nncol],
		  PPARMS parms,
		  PCONTROLS controls,
		  int current, int nbin,
		  int c1, int c2,
		  int global_startrow,
		  double arow[nbin][c2-c1])
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    
    if (current_prow == controls->rank_in_col){
	copysubmat(nncol, (double(*)[])(a[current_lrow]+c1), c2-c1, arow, nbin, c2-c1);
    }
    MP_mybcast(arow,sizeof(double)*(c2-c1)*nbin,
	       current_prow,  controls->col_comm);
}


int startrow_for_update(PPARMS parms, PCONTROLS controls,
			int current, int global_startrow)

{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    return startrow;

}

int MP_update_using_lu(int nnrow, int nncol,
			double a[nnrow][nncol],
			PPARMS parms,
			PCONTROLS controls,
			int current, int nbin,
			int c1, int c2,
			int global_startrow,
			double acol[][nbin],
			double arow[nbin][c2-c1])
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;

    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }

    BEGIN_TIMER(timer);
    mydgemm(nrow-startrow, c2-c1, nbin, -1.0, &(acol[0][0]), nbin,
	    &(arow[0][0]), c2-c1, 1.0, &(a[startrow][c1]), nncol );
    END_TIMER(timer,19,((double)(nrow-startrow))*(c2-c1)*nbin*2);

}



int MP_update_multiple_blocked_global(int nnrow, int nncol,
				      double a[nnrow][nncol],
				      PPARMS parms,  PCONTROLS controls,
				      int current, int nbin,
				      int c1, int c2,
				      int global_startrow,
				      int singlecol)
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    double arow[nbin][c2-c1];
    double acol[nnrow][nbin];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	for (i=0;i<nrow-startrow;i++)
	    for(j=0;j<nbin;j++)
		acol[i][j]=a[i+startrow][current_lcol+j];
    }
    if (! singlecol){
	MPI_Bcast(acol,sizeof(double)*(nrow-startrow)*nbin,MPI_BYTE,
		  current_pcol,  controls->row_comm);
    }
    if (current_prow == controls->rank_in_col){
	for (i=0;i<c2-c1;i++)
	    for(j=0;j<nbin;j++)
		arow[j][i]=a[current_lrow+j][i+c1];
    }
    MPI_Bcast(arow,sizeof(double)*(c2-c1)*nbin,MPI_BYTE,
	      current_prow,  controls->col_comm);
    mydgemm(nrow-startrow, c2-c1, nbin, -1.0, &(acol[0][0]), nbin,
	    &(arow[0][0]), c2-c1, 1.0, &(a[startrow][c1]), nncol );

}
int MP_update_multiple_blocked_global_withacol(int nnrow, int nncol,
					       double a[nnrow][nncol],
					       PPARMS parms,
					       PCONTROLS controls,
					       int current, int nbin,
					       double acol[nnrow][nbin],
					       int c1, int c2,
					       int global_startrow,
					       int singlecol)
{
    int current_prow;
    int current_lrow;
    int gs_prow;
    int gs_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,k;
    double arow[nbin][c2-c1] __attribute__((aligned(128)));
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    BEGIN_TIMER(timer0);
    
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_rows(global_startrow, &gs_prow, &gs_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(current, &current_pcol, &current_lcol,
				       parms, controls);
    int startrow;
    int nb = parms->nb;
    if (gs_prow > controls->rank_in_col){
	startrow = (gs_lrow/nb+1)*nb;
    }else if (gs_prow == controls->rank_in_col){
	startrow =gs_lrow;
    }else{
	startrow = (gs_lrow/nb)*nb;
    }
    if (current_pcol == controls->rank_in_row){
	copysubmat(nncol, (double(*)[])(a[startrow]+current_lcol), nbin, acol,
		   nrow-startrow, nbin);
    }
    if (! singlecol){
	MPI_Bcast(acol,sizeof(double)*(nrow-startrow)*nbin,MPI_BYTE,
		  current_pcol,  controls->row_comm);
    }
    if (current_prow == controls->rank_in_col){
	copysubmat(nncol, (double(*)[])(a[current_lrow]+c1), c2-c1, arow,
		   nbin,c2-c1);
    }
    MPI_Bcast(arow,sizeof(double)*(c2-c1)*nbin,MPI_BYTE,
	      current_prow,  controls->col_comm);
    END_TIMER(timer0,27,((double)(nrow-startrow))*(c2-c1));
    BEGIN_TIMER(timer);
    mydgemm(nrow-startrow, c2-c1, nbin, -1.0, &(acol[0][0]), nbin,
	    &(arow[0][0]), c2-c1, 1.0, &(a[startrow][c1]), nncol );
    if (nbin == 8){
	END_TIMER(timer,13,((double)(nrow-startrow))*(c2-c1)*nbin*2);
    }else if (nbin == 16){
	END_TIMER(timer,20,((double)(nrow-startrow))*(c2-c1)*nbin*2);
    }else if (nbin == 32){
	END_TIMER(timer,21,((double)(nrow-startrow))*(c2-c1)*nbin*2);
    }else if (nbin == 64){
	END_TIMER(timer,22,((double)(nrow-startrow))*(c2-c1)*nbin*2);
    }else if (nbin == 128){
	END_TIMER(timer,23,((double)(nrow-startrow))*(c2-c1)*nbin*2);
    }
    END_TIMER(timer,26,((double)(nrow-startrow))*(c2-c1)*nbin*2);

}

static void MP_solve_triangle_for_unit_mat_internal(int nncol,
				 double a[][nncol],
				 int nb,
				 double b[][nb],
				 int m)
{
    int i,ii,j,k;
    for(ii=0;ii<m;ii++){
	for(j=ii+1;j<m;j++){
	    v2df acopy = (v2df){-a[j][ii],-a[j][ii]};
	    v2df* src = (v2df*)b[ii];
	    v2df* dest = (v2df*)b[j];
	    for(k=0;k<j/2;k++)dest[k] += acopy*src[k];
	    if(j&1)		b[j][j-1] -= a[j][ii]*b[ii][j-1];
	}
    }
}

static void MP_solve_triangle_for_unit_mat_internal_omp(int nncol,
				 double a[][nncol],
				 int nb,
				 double b[][nb],
				 int m)
{
    int i,ii,j;
    for(ii=0;ii<m;ii++){
#pragma omp parallel for private(j)
	for(j=ii+1;j<m;j++){
	    int k;
	    v2df acopy = (v2df){-a[j][ii],-a[j][ii]};
	    v2df* src = (v2df*)b[ii];
	    v2df* dest = (v2df*)b[j];
	    for(k=0;k<j/2;k++)dest[k] += acopy*src[k];
	    if(j&1)		b[j][j-1] -= a[j][ii]*b[ii][j-1];
	}
    }
}

void MP_solve_triangle_for_unit_mat(int nncol,
				    double a[][nncol],
				    int nb,
				    double b[][nb],
				    int m);


static void MP_solve_triangle_for_unit_mat_recursive(int nncol,
				 double a[][nncol],
				 int nb,
				 double b[][nb],
				 int m)
{
    int i,ii,j,k;
    if (m < 128){
	MP_solve_triangle_for_unit_mat_internal(nncol, a, nb, b,m);
	return;
    }
    const int mhalf = m/2;
    MP_solve_triangle_for_unit_mat_recursive(nncol, a, nb, b,mhalf);

    mydgemm( mhalf, mhalf, mhalf, -1.0, &(a[mhalf][0]), nncol,
		 &(b[0][0]), nb, 1.0, &(b[mhalf][0]),nb );

    double bwork[mhalf][mhalf] __attribute__((aligned(128)));
    double bwork2[mhalf][mhalf] __attribute__((aligned(128)));
    for (j=0;j<mhalf;j++)
	for (k=0;k<mhalf;k++)bwork[j][k]=0.0;
    for (j=0;j<mhalf;j++)bwork[j][j]=1.0;
    MP_solve_triangle_for_unit_mat_recursive(nncol,
					     (double(*)[])(&a[mhalf][mhalf]),
					     mhalf, bwork,mhalf);
    for(i=0;i<mhalf;i++)
	for(j=0;j<mhalf;j++)
	    bwork2[i][j]=b[i+mhalf][j];

    mydgemm(mhalf, mhalf, mhalf, 1.0, (double*)bwork,mhalf,
	    (double*)bwork2, mhalf, 0.0, &(b[mhalf][0]),nb );

    for (j=0;j<mhalf;j++)
	for (k=0;k<j+1;k++)b[mhalf+j][mhalf+k]=bwork[j][k];
}    

void MP_solve_triangle_for_unit_mat(int nncol,
				    double a[][nncol],
				    int nb,
				    double b[nb][nb],
				    int m)

{
    int ii,j,k;
    BEGIN_TIMER(timer);
    for (j=0;j<nb;j++)
	for (k=0;k<nb;k++)b[j][k]=0.0;
    for (j=0;j<nb;j++)b[j][j]=1.0;

    MP_solve_triangle_for_unit_mat_recursive(nncol, (double(*)[])  (&a[0][0]),
					 nb, b, m);
    END_TIMER(timer,35,((double)(nb))*nb*nb);
}    

int MP_solve_triangle(int nncol, double a[][nncol], 
		      int ncols, int m,
		      double acol[m][m])
{
    int i,ii,j,k;
    //    current =ii
    //   c0=i+m
    //   c1=iend
    //   r0=ii+1
    //   r1 = i+m
    double b[m][m] __attribute__((aligned(128)));
    double awork[m][ncols] __attribute__((aligned(128)));
    MP_solve_triangle_for_unit_mat(m,acol,m,b,m);
    for(j=0;j<m;j++){
	for (k=0;k<ncols;k++){
	    awork[j][k]=a[j][k];
	    a[j][k]=0;
	}
    }
#if 1
    mydgemm(m, ncols, m, 1.0, &(b[0][0]), m,
		 &(awork[0][0]), ncols, 0.0, &(a[0][0]), nncol );
#else
    for (k=0;k<ncols;k++){
	for(j=0;j<m;j++)
	    for(ii=0;ii<m;ii++)
		a[j][k]+=  b[j][ii]*awork[ii][k];
    }
#endif
}

int MP_solve_triangle_using_inverse(int nncol, double a[][nncol], 
				    int ncols, int m,
				    double b[m][m])
{
    int i,ii,j,k;
    double awork[m][ncols] __attribute__((aligned(128)));
    v2df zerov=(v2df){0.0,0.0};
    if (ncols<32){
	for(j=0;j<m;j++){
	    for (k=0;k<ncols;k++){
		awork[j][k]=a[j][k];
		a[j][k]=0;
	    }
	}
    }else{
	copysubmat(nncol,a,ncols,awork,m,ncols);
    }

    mydgemm(m, ncols, m, 1.0, &(b[0][0]), m,
		 &(awork[0][0]), ncols, 0.0, &(a[0][0]), nncol );
}
int MP_calculate_ld_old(int m, double a[][m], 
		    int ncols, double b[m][m],
		    int current,
		    PCONTROLS controls,  PPARMS parms)
{
    int i,ii,j,k;
    int current_prow, current_lrow;
    double awork[ncols][m];
    for(j=0;j<ncols;j++){
	for (k=0;k<m;k++){
	    awork[j][k]=a[j][k];
	    a[j][k]=0;
	}
    }
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    MPI_Bcast(b,sizeof(double)*m*m,MPI_BYTE, current_prow,
	      controls->col_comm);
#if 1
    mydgemm(ncols, m, m, 1.0, &(awork[0][0]), m,
		 &(b[0][0]), m, 0.0, &(a[0][0]), m );
#else
    for (k=0;k<m;k++){
	for(j=0;j<ncols;j++)
	    for(ii=0;ii<m;ii++)
		a[j][k]+=  awork[j][ii]*b[ii][k];
    }
#endif
}

int MP_calculate_ld_phase1(int m, double a[][m], 
		    int ncols, double b[m][m],
		    int current,
		    PCONTROLS controls,  PPARMS parms)
// broadcast diagnal panel (already triangle)
// so that it can then be multiplied with L panel    
{
    int i,ii,j,k;
    int current_prow, current_lrow;
    BEGIN_TIMER(timer);
    convert_global_index_to_local_rows(current, &current_prow, &current_lrow,
				       parms, controls);
    MPI_Bcast(b,sizeof(double)*m*m,MPI_BYTE, current_prow,
	      controls->col_comm);
    END_TIMER(timer,38,(m+0.0)*m);
}

int MP_calculate_ld_phase2(int m, double a[][m], 
		    int ncols,  double awork[][m], double b[m][m],
		    int current,
		    PCONTROLS controls,  PPARMS parms)
{
    int i,ii,j,k;
    int current_prow, current_lrow;
    BEGIN_TIMER(timer);
    for(j=0;j<ncols;j++){
	for (k=0;k<m;k++){
	    awork[j][k]=a[j][k];
	    a[j][k]=0;
	}
    }
    mydgemm(ncols, m, m, 1.0, &(awork[0][0]), m,
		 &(b[0][0]), m, 0.0, &(a[0][0]), m );
    END_TIMER(timer,6,((double)(ncols))*m*m*2);
}

int MP_calculate_ld(int m, double a[][m], 
		    int ncols, double awork[][m],  double b[m][m],
		    int current,
		    PCONTROLS controls,  PPARMS parms)
{

    MP_calculate_ld_phase1(m, a, ncols, b, current, controls, parms);
    MP_calculate_ld_phase2(m, a, ncols, awork, b, current, controls, parms);
}

int MP_update_multiple_using_diagonal(int nnrow, int nncol,
				     double a[nnrow][nncol],
				     PPARMS parms,  PCONTROLS controls,
				      int firstrow, int c1, int c2,int nrows,
				      double acolinv[nrows][nrows])

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,ii;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(firstrow, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(firstrow, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	MP_solve_triangle_using_inverse(nncol,
					(double(*)[])(&a[current_lrow][c1]),
					c2-c1,
					nrows,acolinv);
    }
}

int MP_store_diagonal_inverse(int nnrow, int nb,
				     double dinv[nnrow][nb],
				     PPARMS parms,  PCONTROLS controls,
				      int firstrow,  double acolinv[nb][nb])

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,ii;
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(firstrow, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(firstrow, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	for (i=0;i<nb;i++){
	    for(j=0;j<nb;j++){
		dinv[current_lrow+i][j]=acolinv[i][j];
	    }
	}
    }
}

int MP_process_diagonal(int nnrow, int nncol,
			double a[nnrow][nncol],
			PPARMS parms,  PCONTROLS controls,
			int firstrow, int nrows,
			double acolinv[nrows][nrows],
			int singlecol)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,ii;
    double acol[nrows][nrows];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(firstrow, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(firstrow, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	if (current_pcol == controls->rank_in_row){
	    int endrow;
	    int nrhalf = nrows/2;
	    endrow = current_lrow+nrows;
	    for (i=0;i<endrow-current_lrow;i++){
		v2df * src = (v2df*)&(a[i+current_lrow][current_lcol]);
		v2df *dest = (v2df*)&(acol[i][0]);
		for (ii=0;ii<nrhalf;ii++){
		    //   acol[i][ii]=a[i+current_lrow][current_lcol+ii];
		    dest[ii]=src[ii];
		    // need to be fixed
		}
	    }
	}
	if (current_pcol == controls->rank_in_row){
#if 0	    
	    dprintf("update_local firstrow=%d\n", firstrow);
	    for(i=0;i<nrows;i++){
		fprintf(stderr, "i= %2d:", i);
		for(j=0;j<nrows;j++){
		    fprintf(stderr, " %10.3e",acol[i][j]);
		}
		fprintf(stderr,"\n");
	    }
#endif	    
	    MP_solve_triangle_for_unit_mat(nrows,acol,nrows,acolinv,nrows);
	}
	if (! singlecol){
	    MPI_Bcast(acolinv,sizeof(double)*nrows*nrows,MPI_BYTE,
		      current_pcol,  controls->row_comm);
	}
    }
}

int MP_process_diagonal_phase1(int nnrow, int nncol,
			double a[nnrow][nncol],
			PPARMS parms,  PCONTROLS controls,
			int firstrow, int nrows,
			double acolinv[nrows][nrows],
			int singlecol)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,ii;
    double acol[nrows][nrows] __attribute__((aligned(128)));
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(firstrow, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(firstrow, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	if (current_pcol == controls->rank_in_row){
	    copysubmat(nncol, (double(*)[])(a[current_lrow]+current_lcol), nrows, acol,
		       nrows, nrows);
	    MP_solve_triangle_for_unit_mat(nrows,acol,nrows,acolinv,nrows);
	}
    }
}
int MP_process_diagonal_phase2(int nnrow, int nncol,
			double a[nnrow][nncol],
			PPARMS parms,  PCONTROLS controls,
			int firstrow, int nrows,
			double acolinv[nrows][nrows],
			int singlecol)

{
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    int i,j,ii;
    double acol[nrows][nrows];
    int ncol = controls->ncol+1;
    int nrow = controls->nrow;
    
    convert_global_index_to_local_rows(firstrow, &current_prow, &current_lrow,
				       parms, controls);
    convert_global_index_to_local_cols(firstrow, &current_pcol, &current_lcol,
				       parms, controls);
    if (current_prow == controls->rank_in_col){
	if (! singlecol){
	    MPI_Bcast(acolinv,sizeof(double)*nrows*nrows,MPI_BYTE,
		      current_pcol,  controls->row_comm);
	}
    }
}

void backward_sub_mpi(int nnrow, int nncol, double a[nnrow][nncol],double b[],
		      PCONTROLS controls,   PPARMS parms)
{
    int i,j,k;
    int previous_pcol = -1;
    int n = controls->nrow*parms->nprow;
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    MPI_Status status; 
    for (i=0;i<controls->nrow;i++)b[i] = a[i][controls->ncol];
    for(i=n-1;i>=1;i--){
	convert_global_index_to_local_rows(i, &current_prow, &current_lrow,
					   parms, controls);
	convert_global_index_to_local_cols(i, &current_pcol, &current_lcol,
				       parms, controls);
	if (previous_pcol == -1) previous_pcol = current_pcol;
	if (current_pcol != previous_pcol){
	    // process column changed, current column should
	    // receive b from previous column
	    if(controls->rank_in_row == previous_pcol){
		MPI_Send(b, controls->nrow, MPI_DOUBLE, current_pcol,
			 MPBSTAG,controls->row_comm);
	    }else if (controls->rank_in_row == current_pcol){
		MPI_Recv(b, controls->nrow, MPI_DOUBLE, previous_pcol,
			 MPBSTAG,controls->row_comm, &status);
	    }
	}
	if(controls->rank_in_row == current_pcol){
	    // have the current column of a
	    double btmp = b[current_lrow];
	    MPI_Bcast(&btmp,1,MPI_DOUBLE,current_prow,  controls->col_comm);
	    int jmax;
	    int nb = parms->nb;
	    if (controls->rank_in_col == current_prow){
		jmax = current_lrow;
	    }else if (controls->rank_in_col < current_prow){
		jmax = ((current_lrow/nb)+1)*nb;
	    }
	    else{
		jmax = (current_lrow/nb)*nb;
	    }
	    for(j=0;j<jmax;j++){
		b[j] -= btmp*a[j][current_lcol];
	    }
	}
	previous_pcol=current_pcol;
    }
    MPI_Bcast(b,controls->nrow,MPI_DOUBLE,
	      current_pcol,  controls->row_comm);
    for (i=0;i<controls->nrow;i++)a[i][controls->ncol] = b[i];

}

void backward_sub_blocked_mpi(int nnrow, int nncol, double a[nnrow][nncol],
			      int nb, double dinv[nnrow][nb],
			      double b[], PCONTROLS controls,   PPARMS parms)
{
    int i,ii,j,k;
    int previous_pcol = -1;
    int n = controls->nrow*parms->nprow;
    int current_prow;
    int current_lrow;
    int current_pcol;
    int current_lcol;
    MPI_Status status;
    //    dprintf(9,"enter backward_sub_blocked_mpi\n");
    for (i=0;i<controls->nrow;i++)b[i] = a[i][controls->ncol];
    for(ii=n-1;ii>=1;ii-=nb){
	//	dprintf(9,"ii=%d\n", ii);
	convert_global_index_to_local_cols(ii, &current_pcol, &current_lcol,
					   parms, controls);
	convert_global_index_to_local_rows(ii, &current_prow, &current_lrow,
					   parms, controls);

	if (previous_pcol == -1) previous_pcol = current_pcol;
	if (parms->npcol > 1){
	    if (ii != n-1){
		if(controls->rank_in_row == previous_pcol){
		    MPI_Send(b, controls->nrow, MPI_DOUBLE, current_pcol,
			     MPBSTAG,controls->row_comm);
		}else if (controls->rank_in_row == current_pcol){
		    MPI_Recv(b, controls->nrow, MPI_DOUBLE, previous_pcol,
			     MPBSTAG,controls->row_comm, &status);
		}
	    }
	}
	//	dprintf(9,"send/receive end ii=%d\n", ii);
	if(controls->rank_in_row == current_pcol){
	    double bwork[nb];
	    int jend;
	    if  (controls->rank_in_col == current_prow){
		int jmin = current_lrow - nb+1;
#if 0		
		for(k=0;k<nb;k++){
		    int jmax;
		    jmax = current_lrow-k;
		    bwork[k]=b[jmax];
		    double btmp=bwork[k];
		    for(j=jmin;j<jmax;j++){
			b[j] -= btmp*a[j][current_lcol-k];
		    }
		}
#endif
#if 0
		for(j=current_lrow-1;j>=current_lrow-nb+1;j--){
		    int r0 = current_lrow-j+1;
		    int c0 = current_lcol-j+1;
		    for(k=0;k<current_lrow-j;k++){
			b[j] -= b[current_lrow-k]*a[j][current_lcol-k];
		    }
		}
#endif
#if 1		
		for(j=current_lrow-1;j>=current_lrow-nb+1;j--){
		    int r0 = j+1;
		    int c0 = current_lcol-current_lrow+r0;
		    int kmax = current_lrow-j;
		    double btmp = b[j];
		    double * bp = b+r0;
		    double * ap = &(a[j][c0]);
		    //		    double * apn = &(a[j-1][c0]);
		    //		    for (k=0;k<kmax; k+= 16)__builtin_prefetch(apn+k,0,0);
		    for(k=0;k<kmax;k++){
			btmp -= bp[k]*ap[k];
		    }
		    b[j]=btmp;
		}
#endif		
		for(k=0;k<nb;k++){
		    int jmax;
		    jmax = current_lrow-k;
		    bwork[k]=b[jmax];
		}
		//		for(j=jmin;j<=current_lrow;j++){
		//		    dprintf(9,"ad[%d] = %10.4g %10.4g\n", j,a[j][current_lcol-1],
		//			    a[j][current_lcol]);
		//		}
		jend = jmin;
		if (parms->nprow > 1)
		    MPI_Bcast(bwork,nb,MPI_DOUBLE,current_prow,controls->col_comm);
		//		dprintf(9,"source bcast end ii=%d\n", ii);
	    }else{
		if (parms->nprow > 1)
		    MPI_Bcast(bwork,nb,MPI_DOUBLE,current_prow,controls->col_comm);
		//		dprintf(9,"dest bcast end ii=%d\n", ii);
		if (controls->rank_in_col < current_prow){
		    jend = ((current_lrow/nb)+1)*nb;
		}
		else{
		    jend = (current_lrow/nb)*nb;
		}
	    }
	    double bwork2[jend];
	    double bwork3[jend];
#if 0	    
	    for(j=0;j<jend;j++){
		bwork2[j]=0;
		for(k=0;k<nb;k++){
		    bwork2[j] += bwork[k]*a[j][current_lcol-k];
		}
	    }
#endif

	    double bwork4[nb];
	    for(k=0;k<nb;k++) bwork4[k]=bwork[nb-1-k];
#pragma omp parallel for private(j,k)
	    for(j=0;j<jend;j++){
		double btmp=0;
		double *ap=&(a[j][current_lcol-nb+1]);
		for(k=0;k<nb;k++){
		    btmp += bwork4[k]*ap[k];
		}
		bwork2[j]=btmp;
	    }
	    
#define	DUPOSTMUL    
#ifndef DUPOSTMUL
	    for(j=0;j<jend;j++)	b[j] -= bwork2[j];
#else
#pragma omp parallel for private(j,k) schedule(static)
	    for(j=0;j<jend;j++){
		bwork3[j]=0;
		int jmin = (j /nb)*nb;
		for(k=0;k<nb;k++){
		    bwork3[j] += bwork2[k+jmin]*dinv[j][k];
		}
	    }
	    for(j=0;j<jend;j++)	b[j] -= bwork3[j];
#endif	    
	    //	    dprintf(9,"calc end ii=%d\n", ii);
	}
	previous_pcol=current_pcol;
    }

    if (parms->npcol > 1)
	MPI_Bcast(b,controls->nrow,MPI_DOUBLE, current_pcol,  controls->row_comm);
    for (i=0;i<controls->nrow;i++)a[i][controls->ncol] = b[i];
    //    dprintf(9,"backward_sub all t end \n");

}

void check_solution_mpi(int nnrow, int nncol, double a[nnrow][nncol],
			double b[], double bcopy[],
			PCONTROLS controls,   PPARMS parms)
{
    int i,j,k;
    int nrow = controls->nrow;
    int ncol=controls->ncol;
    MPI_Allgather(b,nrow,MPI_DOUBLE,bcopy,nrow,MPI_DOUBLE, controls->col_comm);
    double bcopy2[ncol];
    for(i=0;i<ncol;i++){
	int iglobal=global_colid(i, parms,controls);
	int rowpid, rowlid;
	convert_global_index_to_local_rows(iglobal, &rowpid, &rowlid,
					   parms, controls);
	int index= nrow*rowpid+rowlid;
	bcopy2[i]=bcopy[index];
	//	dprintf(9,"b[%d]=%g %g\n", i, bcopy2[i], bcopy[i]);
    }
    for(i=0;i<nrow;i++){
	double btmp=0;
	for(j=0;j<ncol;j++) {
	    btmp+= a[i][j]*bcopy2[j];
	    //	    dprintf(9,"a[%d][%d]=%g, bcopy2=%g %g\n", i, j,
	    //		    a[i][j], bcopy2[j],btmp);
	}
	
	bcopy[i]=btmp;
    }
    MPI_Allreduce(bcopy,b,nrow,MPI_DOUBLE, MPI_SUM, controls->row_comm);
    for (i=0;i<controls->nrow;i++)a[i][controls->ncol] = b[i];
    
}

double print_errors(double b[], double b0[], PCONTROLS controls,   PPARMS parms)
{
    int i,j,k;
    int nrow=controls->nrow;
    double esum = 0;
    double emax = 0;
    for (i=0;i<nrow;i++){
	double err = b[i]-b0[i];
	esum+= err*err;
	if (emax < fabs(err)) emax = fabs(err);
    }
    dprintf(9," esum = %e\n", esum);
    double totalerror=0;
    double totalemax=0;
    MPI_Reduce(&esum,&totalerror, 1, MPI_DOUBLE, MPI_SUM,
	       0, controls->col_comm);
    MPI_Reduce(&emax,&totalemax, 1, MPI_DOUBLE, MPI_MAX,
	       0, controls->col_comm);
    if (controls->rank_in_col == 0){
	dprintf(0,"Errors = %e %e %e %e\n\n",  sqrt(totalerror), totalemax, esum, emax);
	printf("\n Error = %e %e\n",  sqrt(totalerror), totalemax);
    }
    return totalemax;
}

double Mmax(double a, double b)
{
    if (a>b){
	return a;
    }else{
	return b;
    }
}

void HPLlogprint(FILE * f, int N, int NB, int nprow, int npcol, double elapsed)
{
    fprintf(f, "%s%s\n",
	    "======================================",
	    "======================================" );
    fprintf(f,"%s%s\n",
	   "T/V                N    NB     P     Q",
	   "               Time             Gflops" );
    fprintf(f, "%s%s\n",
	    "--------------------------------------",
	    "--------------------------------------" );
    double Gflops = ( ( (double)(N) /   1.0e+9 ) * 
		      ( (double)(N) / elapsed ) ) * 
	( ( 2.0 / 3.0 ) * (double)(N) + ( 3.0 / 2.0 ) );
    
    fprintf(f,"W%c%1d%c%c%1d%c%1d%12d %5d %5d %5d %18.2f %18.3e\n",
	   'R',0,'1', 'R', 2,  'C', 32,
	   N, NB, nprow, npcol, elapsed, Gflops );
}

void HPLerrorprint(FILE * f, double emax, double a1, double ao,
		   double b1, double bo, double n)
{
    double epsil =2e-16;
    double thrsh = 16;
    double resid1 = emax / (epsil * a1 * n );
    double resid2 = emax / (epsil * a1 * b1 );
    double resid3 = emax / (epsil * ao * bo * n);
    //   fprintf(f,"err= %e Anorms= %e %e Xnorms= %e %e\n",resid0, a1,  ao, b1, bo);
    int kpass =0;
    int kfail= 0;
    if( ( Mmax( resid1, resid2 ) < thrsh ) &&
       ( resid3 < thrsh ) ) (kpass)++;
   else                           (kfail)++;

    fprintf(f,"%s%s\n",
	   "--------------------------------------",
	   "--------------------------------------" );
    fprintf(f,"%s%16.7f%s%s\n",
	   "||Ax-b||_oo / ( eps * ||A||_1  * N        ) = ", resid1,
	   " ...... ", ( resid1 < thrsh ? "PASSED" : "FAILED" ) );
    fprintf(f,"%s%16.7f%s%s\n",
	    "||Ax-b||_oo / ( eps * ||A||_1  * ||x||_1  ) = ", resid2,
	    " ...... ", ( resid2 < thrsh ? "PASSED" : "FAILED" ) );
    fprintf(f,"%s%16.7f%s%s\n",
	   "||Ax-b||_oo / ( eps * ||A||_oo * ||x||_oo ) = ", resid3,
	   " ...... ", ( resid3 < thrsh ? "PASSED" : "FAILED" ) );
    
    if( ( resid1 >= thrsh ) || ( resid2 >= thrsh ) ||
	( resid3 >= thrsh ) ) {
	fprintf(f,"%s%18.6f\n",
	       "||Ax-b||_oo  . . . . . . . . . . . . . . . . . = ", emax );
	fprintf(f,"%s%18.6f\n",
	       "||A||_oo . . . . . . . . . . . . . . . . . . . = ", ao );
	fprintf(f, "%s%18.6f\n",
		"||A||_1  . . . . . . . . . . . . . . . . . . . = ", a1 );
	fprintf(f,"%s%18.6f\n",
	       "||x||_oo . . . . . . . . . . . . . . . . . . . = ", bo );
	fprintf(f,"%s%18.6f\n",
	       "||x||_1  . . . . . . . . . . . . . . . . . . . = ", b1 );
    }
    fprintf(f, "%s%s\n",
	    "======================================",
	    "======================================" );
    fprintf(f, "\n%s %6d %s\n", "Finished", 1,
	    "tests with the following results:" );
    fprintf(f,"         %6d %s\n", kpass,
                      "tests completed and passed residual checks," );
    fprintf(f,"         %6d %s\n", kfail,
	   "tests completed and failed residual checks," );
    fprintf(f,"         %6d %s\n", 0,
                      "tests skipped because of illegal input values." );
    fprintf(f,"%s%s\n",
	   "--------------------------------------",
	   "--------------------------------------" );
    fprintf(f,"\nEnd of Tests.\n" );
    fprintf(f,"%s%s\n",
	   "======================================",
	   "======================================" );
}


void printmat_MP(int nnrow, int nncol, double a[nnrow][nncol], 
	    PCONTROLS controls,  PPARMS parms)
{
    int procid =MP_proccount();
    int ii, i, j;
    MPI_Barrier(MPI_COMM_WORLD);
    for (ii=0;ii<procid; ii++){
	//	fprintf(stderr,"printmatmp, ids = %d %d\n", ii, MP_myprocid());
	if (MP_myprocid() == ii){
	    fprintf(stderr,"Printmat_MP: Proc %d, loc = %d %d\n",MP_myprocid(),
		   controls->rank_in_row, controls->rank_in_col);
	    for(i=0;i<controls->nrow;i++){
		fprintf(stderr,"%3d: ", i);
		for(j=0;j<controls->ncol+1;j++) fprintf(stderr," %10.3e", a[i][j]);
		fprintf(stderr,"\n");
	    }
	    fprintf(stderr,"\n");
	    
	}
	MPI_Barrier(MPI_COMM_WORLD);
    }
}

void lu_forward_onestep_mpi(int i,int nnrow, int nncol, double a[nnrow][nncol],
			    double b[],  PCONTROLS controls,  PPARMS parms)
{
    //    dprintf(9,"lu_mpi enter i=%d\n", i);
    int imax= MP_find_pivot(nnrow, nncol, a, parms, controls, i);
    MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,imax,0,nncol);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
    //    dprintf(9,"lu_mpi after swap_rows i=%d\n", i);
    MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,0,nncol,0);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
    //    dprintf(9,"lu_mpi after scale_rows i=%d\n", i);
    MP_update_single(nnrow, nncol, a, parms, controls,i);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
    //    dprintf(9,"lu_mpi after update_single i=%d\n", i);
}

void column_decomposition_mpi(int ifirst,int nb,int nnrow, int nncol,
			      double a[nnrow][nncol],
			      double b[],  int pv[],
			      PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    int c1, c2;
    //    dprintf(9,"column_decomposition_mpi  ifirst=%d nb=%d\n",ifirst,nb);
    convert_global_col_range_to_local_range(ifirst, ifirst+nb-1,&c1, &c2,
					    parms, controls);
    c2 ++;

    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	int imax= MP_find_pivot(nnrow, nncol, a, parms, controls, i);
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,imax,c1,c2);
	pv[ii]=imax;
	MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,1);
	int cu1, cu2;
	convert_global_col_range_to_local_range(i+1, ifirst+nb-1,&cu1, &cu2,
						parms, controls);
	cu2 ++;
	MP_update_single_blocked(nnrow, nncol, a, parms, controls,i,cu1,cu2);
    }
}
void print2dmat_MP(int nnrow, int nncol, double a[nnrow][nncol], char* s)
{
    int ii, i, j;
    fprintf(stderr,"Print2dmat_MP: Proc %d, name=%s \n",MP_myprocid(), s);
    for(i=0;i<nnrow;i++){
	fprintf(stderr,"%3d: ", i);
	for(j=0;j<nncol;j++) fprintf(stderr," %10.3e", a[i][j]);
	fprintf(stderr,"\n");
    }
    fprintf(stderr,"\n");
    
}


void column_decomposition_mpi_transposed(int ifirst,int nb,int nnrow, int nncol,
			      double a[nncol][nnrow],
			      double b[],  int pv[],
			      PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    int c1, c2;
    double arow[nncol];
    double acol[nnrow];

    //    dprintf(9,"column_decomposition_mpi  ifirst=%d nb=%d\n",ifirst,nb);
    convert_global_col_range_to_local_range(ifirst, ifirst+nb-1,&c1, &c2,
					    parms, controls);
    c1 %= nb;
    c2 %= nb;
    c2 ++;

    int havec=have_current_col(ifirst,  parms, controls);

    if (havec){
	//	print2dmat_MP(nb, nnrow,  a, "transposed a");
	for (ii=0;ii<nb;ii++){
	    i = ii+ifirst;
	    //	    dprintf(9,"coldec trans i=%d\n", i);
	    int imax= MP_find_pivot_transposed(nnrow, nb, a, parms, controls, i);
	    //	    dprintf(9,"coldec trans i=%d imax=%d find pivot end\n", i,imax);
	    MP_swap_rows_blocked_transposed(nnrow, nb, a, parms, controls,i,imax,c1,c2);
	    //	    dprintf(9,"coldec trans i=%d swap rows end\n", i);
	    pv[ii]=imax;
	    //	    print2dmat_MP(nb, nnrow,  a, "after swap");
	    MP_scale_row_blocked_transposed(nnrow, nb, a, parms, controls,i,c1,c2);
	    //	    print2dmat_MP(nb, nnrow,  a, "after scale");
	    //	    dprintf(9,"coldec trans i=%d scale rows end\n", i);
	    
	    int cu1, cu2;
	    convert_global_col_range_to_local_range(i+1, ifirst+nb-1,&cu1, &cu2,
						    parms, controls);
	    cu1 %= nb;
	    cu2 %= nb;
	    cu2 ++;
	    if(ii<nb-1){
		//		dprintf(9,"coldec trans i=%d call update\n", i);
		MP_update_single_blocked_transposed(nnrow, nb, a,
						    parms, controls,
						    i,cu1,cu2,arow,acol);
	    }
	    //	    dprintf(9,"coldec trans i=%d update end\n", i);
	}
    }
}
void transpose_rowtocol8(int nnrow, int nncol,
			   double a[nnrow][nncol], double at[][nnrow],
			   int istart)
{
    int i,j,k;
    const int m=8;
    int mend;
#pragma omp parallel for private(i,j,k)   schedule(static)
    for(i=istart;i<nnrow;i+=m){
	double atmp[m][m] __attribute__((aligned(128)));
	//	BEGIN_TSC;
	for(k=0;k<m;k++){
	    v2df * ak = (v2df*) a[i+k];
	    v2df * akk = (v2df*) atmp[k];
	    asm("prefetchnta %0"::"m"(a[i+k+m*2][0]):"memory");
	    //	    asm("prefetchnta %0"::"m"(a[i+k+m*2][8]):"memory");
	    //	    __builtin_prefetch(a[i+k+m*2],0,0);
	    //	    __builtin_prefetch(a[i+k+m*2]+8,0,0);
	    akk[0]  =ak[0];
	    akk[1]  =ak[1];
	    akk[2]  =ak[2];
	    akk[3]  =ak[3];
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
	}
	//	END_TSC(t2,18);
	//	}			   int istart)
    }

}


void transpose_rowtocol8_0(int nnrow, int nncol,
			   double a[nnrow][nncol], double at[][nnrow],
			   int istart)
{
    int i,j,k;
    const int m=8;
    double atmp[m][m];
    for(i=istart;i<nnrow;i+=m){
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

void MP_transpose_rowtocol(int nnrow, int nncol,
			   double a[nnrow][nncol],int m, double at[][nnrow],
			   int istart)
{
    if (m == 8){
	transpose_rowtocol8(nnrow, nncol, a, at, istart);
	return;
    }
    int i,j,k;
    double atmp[m][m];
    for(i=istart;i<nnrow;i+=m){
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
}

void transpose_coltorow8(int nnrow,int nncol, double a[nnrow][nncol],
			 double at[][nnrow],   int istart)
{
    int i,j,k;
    const int m=8;
    double atmp[m][m];
#pragma omp parallel for private(i,j,k,atmp)	  schedule(static)
    for(i=istart;i<nnrow;i+=m){
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

void MP_transpose_coltorow(int nnrow,int nncol,
			   double a[nnrow][nncol],int m, double at[][nnrow],
			   int istart)
{
    if (m == 8){
	transpose_coltorow8(nnrow, nncol, a, at, istart);
	return;
    }
    int i,j,k;
    double atmp[m][m];
    for(i=istart;i<nnrow;i+=m){
	for(j=0;j<m;j++){
	    double * atj = at[j]+i;
	    for(k=0;k<m;k+=2){
		atmp[k][j]  =atj[k];
		atmp[k+1][j]  =atj[k+1];
		//		atmp[k+2][j]  =atj[k+2];
		//		atmp[k+3][j]  =atj[k+3];
	    }
	}
	for(k=0;k<m;k++){
	    double * aik = a[i+k];
	    for(j=0;j<m;j+=2){
		aik[j] = atmp[k][j];
		aik[j+1] = atmp[k][j+1];
		//		aik[j+2] = atmp[k][j+2];
		//		aik[j+3] = atmp[k][j+3];
	    }
	}
    }
}

void column_decomposition_mpi_with_transpose(int ifirst,int nb,
					     int nnrow, int nncol,
					     double a[nnrow][nncol],
					     double b[],  int pv[],
					     PCONTROLS controls,
					     PPARMS parms)
{
    double awork[nb][nnrow] __attribute__((aligned(128)));
    int c1, c2;
    //    dprintf(9,"column_decomposition_mpi_wt  ifirst=%d nb=%d\n",ifirst,nb);
    convert_global_col_range_to_local_range(ifirst, ifirst+nb-1,&c1, &c2,
					    parms, controls);
    c2 ++;
    int localfirst=first_row(ifirst, parms, controls);
    //    dprintf(9,"column_decomposition_mpi_wt  c1=%d c2=%d lfirst=%d\n",
    //	    c1,c2,localfirst);
    BEGIN_TIMER(timer0);
    MP_transpose_rowtocol(nnrow,nncol, (double(*)[])  (&a[0][c1]),
			  nb, awork,localfirst);
    END_TIMER(timer0,10,((double)nb)*(nnrow-localfirst));
    BEGIN_TIMER(timer1);
    column_decomposition_mpi_transposed(ifirst,nb,nnrow,nncol,
					awork, b, pv,controls,parms);
    END_TIMER(timer1,11,((double)nb)*nb*(nnrow-localfirst));
    BEGIN_TIMER(timer2);
    MP_transpose_coltorow(nnrow,nncol, (double(*)[])  (&a[0][c1]),
			  nb, awork,localfirst);
    END_TIMER(timer2,12,((double)nb)*(nnrow-localfirst));

}

void process_right_part_mpi(int ifirst,int nb,int ncols, int nnrow, int nncol,
			      double a[nnrow][nncol],
			      double b[],  int pv[],
			    PCONTROLS controls,  PPARMS parms,
			    int singlecol)
{
    int i,ii;
    int c1, c2;
    //    dprintf(9,"process_rp ifirst=%d nb=%d\n", ifirst,nb);

    double acolinv[nb][nb];

    MP_process_diagonal(nnrow, nncol, a, parms, controls,
			ifirst,nb,acolinv,singlecol);

    convert_global_col_range_to_local_range(ifirst+nb, ifirst+nb+ncols-1,
					    &c1, &c2,  parms, controls);
    c2++;
    if (ncols <= 0) c2 = nncol;
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,singlecol);
    }
    //    dprintf(9,"process_rp, ifirst=%d\n", ifirst);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
    
    
    //    MP_update_multiple_blocked_local(nnrow, nncol, a, parms, controls,
    //				     ifirst,c1,c2,nb);
    
    MP_update_multiple_using_diagonal(nnrow, nncol, a, parms, controls,
				      ifirst,c1,c2,nb,acolinv);
    MP_update_multiple_blocked_global(nnrow, nncol, a, parms, controls,
				      ifirst,nb,c1,c2,ifirst+nb,1);
}
void process_right_part_mpi_withacol(int ifirst,int nb,int ncols,
				     int nnrow, int nncol,
				     double a[nnrow][nncol],
				     double acol[nnrow][nb],
				     double b[],  int pv[],
				     PCONTROLS controls,  PPARMS parms,
				     int singlecol)
{
    int i,ii;
    int c1, c2;
    //    dprintf(9,"process_rp ifirst=%d nb=%d\n", ifirst,nb);

    double acolinv[nb][nb];
    BEGIN_TIMER(timer2);

    MP_process_diagonal(nnrow, nncol, a, parms, controls,
			ifirst,nb,acolinv,singlecol);

    END_TIMER(timer2,24,((double)nb)*nb*nb/6.0);
    BEGIN_TIMER(timer3);
    convert_global_col_range_to_local_range(ifirst+nb, ifirst+nb+ncols-1,
					    &c1, &c2,  parms, controls);
    c2++;
    if (ncols <= 0) c2 = nncol;
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,singlecol);
    }
    //    dprintf(9,"process_rp, ifirst=%d\n", ifirst);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
    
    
    //    MP_update_multiple_blocked_local(nnrow, nncol, a, parms, controls,
    //				     ifirst,c1,c2,nb);
    
    END_TIMER(timer3,36,((double)nb)*(c2-c1)*2);
    BEGIN_TIMER(timer0);
    MP_update_multiple_using_diagonal(nnrow, nncol, a, parms, controls,
				      ifirst,c1,c2,nb,acolinv);
    END_TIMER(timer0,24,((double)nb)*nb*(c2-c1)*2);
    BEGIN_TIMER(timer1);
    MP_update_multiple_blocked_global_withacol(nnrow, nncol, a, parms, controls,
				      ifirst,nb,acol,c1,c2,ifirst+nb,1);
    END_TIMER(timer1,25,((double)nb)*nb*(c2-c1)*2);
}
void process_right_part_mpi_using_dl_old(int ifirst,int nb,int ncols,
				     int nnrow, int nncol,
				     double a[nnrow][nncol],
				     double acolinv[nb][nb],
				     double acol[nnrow][nb],
				     int pv[],
				     PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    int c1, c2;
    //    dprintf(9,"process_rp ifirst=%d nb=%d\n", ifirst,nb);



    convert_global_col_range_to_local_range(ifirst+nb, ifirst+nb+ncols-1,
					    &c1, &c2,  parms, controls);
    c2++;
    if (ncols <= 0) c2 = nncol;
    
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,0);
    }
    //    dprintf(9,"process_rp, ifirst=%d\n", ifirst);
    //    printmat_MP(nnrow, nncol, a, controls, parms);


    //    MP_update_multiple_blocked_local(nnrow, nncol, a, parms, controls,
    //				     ifirst,c1,c2,nb);
    MP_update_multiple_using_diagonal(nnrow, nncol, a, parms, controls,
					 ifirst,c1,c2,nb,acolinv);
    

    MP_update_multiple_blocked_global_using_lmat(nnrow, nncol, a, parms,
						 controls,ifirst,nb,c1,c2,
						 ifirst+nb,acol);
    //    MP_update_multiple_blocked_global(nnrow, nncol, a, parms, controls,
    //				      ifirst,nb,c1,c2,ifirst+nb,0);
}
void process_right_part_mpi_using_dl(int ifirst,int nb,int c1, int c2,
				     int nnrow, int nncol,
				     double a[nnrow][nncol],
				     double acolinv[nb][nb],
				     double acol[nnrow][nb],
				     int pv[],
				     PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    //    dprintf(9,"process_rp_using_dl, ifirst=%d c1, c2=%d %d\n", ifirst,c1,c2);
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	//	dprintf(9,"process_rp, i=%d\n", i);
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	//	dprintf(9,"process_rp swap rows end, i=%d\n", i);
	MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,0);
    }
    //    dprintf(9,"process_rp, ifirst=%d\n", ifirst);
    //    printmat_MP(nnrow, nncol, a, controls, parms);

    //    dprintf(9,"process_rp, enter DTRSM part c1, c2=%d %d\n",c1, c2);

    //    MP_update_multiple_blocked_local(nnrow, nncol, a, parms, controls,
    //				     ifirst,c1,c2,nb);
    MP_update_multiple_using_diagonal(nnrow, nncol, a, parms, controls,
					 ifirst,c1,c2,nb,acolinv);
    
    //    dprintf(9,"process_rp, enter update part c1, c2=%d %d\n",c1, c2);

    MP_update_multiple_blocked_global_using_lmat(nnrow, nncol, a, parms,
						 controls,ifirst,nb,c1,c2,
						 ifirst+nb,acol);
    //    dprintf(9,"process_rp, update part end\n");
    //    MP_update_multiple_blocked_global(nnrow, nncol, a, parms, controls,
    //				      ifirst,nb,c1,c2,ifirst+nb,0);
    //    dprintf(9,"process_rp_using_dl end, ifirst=%d c1, c2=%d %d\n", ifirst,c1,c2);
}
void process_right_part_mpi_using_dls(int ifirst,int nb,int c1, int c2,
				      int nnrow, int nncol,
				      double a[nnrow][nncol],
				      double acolinv[nb][nb],
				      double acol[nnrow][nb],
				      int pv[],
				      double scale[],
				      PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    //    dprintf(9,"process_rp_using_dls, ifirst=%d c1, c2=%d %d\n", ifirst,c1,c2);
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	MP_scale_row_blocked_using_scale(nnrow, nncol, a, parms,
					 controls,i,c1,c2,0,scale[ii]);
    }
    MP_update_multiple_using_diagonal(nnrow, nncol, a, parms, controls,
					 ifirst,c1,c2,nb,acolinv);
    

    MP_update_multiple_blocked_global_using_lmat(nnrow, nncol, a, parms,
						 controls,ifirst,nb,c1,c2,
						 ifirst+nb,acol);
}
void dump_vector(char * s, int * x, int n)
{
    return;
    fprintf(stderr,"%s:", s);
    int i;
    for(i=0;i<n;i++) fprintf(stderr," %2d", x[i]);
    fprintf(stderr,"\n");
}

int generate_src_and_dst_lists(int ifirst,
				int pv[],
				int src[],
				int dst[],
				int * length,
				PCONTROLS controls,  PPARMS parms)
// creates src + dest list from pivot list
{
    int n =parms->n;
    int nb=parms->nb;
    int work[n];
    int i;
    for(i=ifirst; i<n;i++)work[i]=i;
    for(i=ifirst; i<ifirst+nb;i++){
	int p = pv[i-ifirst];
	int tmp = work[i];
	work[i]=work[p];
	work[p]=tmp;
    }
    //    dump_vector("work", work, n-ifirst);
    int ii=0;
    for(i=ifirst; i<n;i++){
	if ((work[i] != i) || (i<ifirst+nb)){
	    dst[ii]=i;
	    src[ii]=work[i];
	    ii++;
	}
    }
    *length=ii;
    return ii;
}

void copysubvect(double *src, double * dst, int length)
    // assume that the length is multiple of 8 and
    // first location is 16-byte-alligned
{
    int j;
    v2df * s = (v2df*) src;
    v2df * d = (v2df*) dst;
    for(j=0;j<length/2;j+=4){
	d[j]=s[j];
	d[j+1]=s[j+1];
	d[j+2]=s[j+2];
	d[j+3]=s[j+3];
    }	
}
void scalesubvect(double scale, double *src, double * dst, int length)
    // assume that the length is multiple of 8 and
    // first location is 16-byte-alligned
{
    int j;
    for(j=0;j<length;j++){
	//	fprintf(stderr,"j=%d src=%e dest=%e\n", j, src[j], dst[j]);
	dst[j]=src[j]*scale;
    }	
}

void scalesubvect_old(double scale, double *src, double * dst, int length)
    // assume that the length is multiple of 8 and
    // first location is 16-byte-alligned
{
    int j;
    v2df * s = (v2df*) src;
    v2df * d = (v2df*) dst;
    v2df ss = {scale,scale};
    for(j=0;j<length/2;j+=4){
	d[j]=ss*s[j];
	d[j+1]=ss*s[j+1];
	d[j+2]=ss*s[j+2];
	d[j+3]=ss*s[j+3];
    }	
}

void local_swap_using_src_and_dest(int ifirst,int nb,int c1, int c2,
				   int nnrow, int nncol,
				   double a[nnrow][nncol],
				   double arow[nb][c2-c1],
				   double scale[],
				   int src[],
				   int dst[],
				   int length,
				   PCONTROLS controls,  PPARMS parms)
{
    int current_proc;
    int current_lloc;
    int pivot_proc;
    int pivot_lloc;
    int i;
    int j;
    convert_global_index_to_local_rows(ifirst, &current_proc, &current_lloc,
				       parms, controls);
    if(current_proc == controls->rank_in_col){
	printf("current_proc=%d, rank=%d\n",current_proc,
	       controls->rank_in_col);
	copysubmat(nncol, (double(*)[])(a[current_lloc]+c1), c2-c1, arow, nb, c2-c1);
    }
    // the following is only for p= (q=?) 1
#pragma omp parallel for private(i)
    for(i=0;i<length;i++){
	if(dst[i]<ifirst+nb){
	    double * s = a[src[i]]+c1;
	    double sc = scale[dst[i]-ifirst];
	    if (src[i]<ifirst+nb)s=arow[src[i]-ifirst];
	    scalesubvect(sc,s,&(a[dst[i]][c1]),c2-c1);
	}
    }
#pragma omp parallel for private(i)
    for(i=0;i<length;i++){
	if(dst[i]>=ifirst+nb){
	    copysubvect(arow[src[i]-ifirst],&(a[dst[i]][c1]),c2-c1);
	}
    }
}


void global_swap_using_src_and_dest(int ifirst,int nb,int c1, int c2,
				   int nnrow, int nncol,
				   double a[nnrow][nncol],
				   double arow[][c2-c1],
				   double ar2[][c2-c1],
				   double scale[],
				   int src[],
				   int dst[],
				   int length,
				   PCONTROLS controls,  PPARMS parms)
{
    int current_proc;
    int current_lloc;
    int pivot_proc;
    int pivot_lloc;
    int i;
    int j;
    int myp = controls->rank_in_col;
    //    fprintf(stderr,"global_swap ifirst=%d nb=%d c1=%d c2=%d\n",ifirst,nb,c1,c2);
    convert_global_index_to_local_rows(ifirst, &current_proc, &current_lloc,
				       parms, controls);
    // Bcast scale in vertical direction
    // need not be done each time...
    MPI_Bcast(scale, nb, MPI_DOUBLE, current_proc,  controls->col_comm);
    //    fprintf(stderr,"global_swap first Bcast end\n");
    //
    // create the data for bcast
    //
    //  first, create list of lines to send on each processors in a column
    //
    //    dprintf(0,"myp, ncol, nrow, npcol, nprow = %d %d %d %d %d\n",
    //	    myp, nncol, nnrow, parms->npcol, parms->nprow);
    int nlines[parms->nprow];
    int iloc[parms->nprow];
    int startloc[parms->nprow];
    int bdst[length];
    dump_vector("src", src, length);
    dump_vector("dst", dst, length);
    for(i=0;i<parms->nprow;i++)nlines[i]=0;
    for(i=0;i<length;i++){
	if(dst[i]<ifirst+nb){
	    int cp, cl;
	    convert_global_index_to_local_rows(src[i], &cp, &cl, parms, controls);
	    nlines[cp]++;
	}
    }
    startloc[0]=0;
    for(i=0;i<parms->nprow-1;i++)startloc[i+1] = startloc[i]+nlines[i];
    //    dump_vector("nlines", nlines, parms->nprow);
    //    dump_vector("startloc", startloc, parms->nprow);
//
//  now each processor knows where to store its data
//
    int ii =0;
    for(i=0;i<parms->nprow; i++)iloc[i]=0;
    for(i=0;i<length;i++){
	//	fprintf(stderr,"convert_index i=%d length=%d\n", i, length);
	if(dst[i]<ifirst+nb){
	    int cp, cl;
	    convert_global_index_to_local_rows(src[i], &cp, &cl, parms, controls);
	    //	    fprintf(stderr, "i=%d,ii=%d,  dst=%d, src=%d, cp=%d, cl=%d\n",
	    //		    i, ii, src[i], dst[i], cp, cl);
	    bdst[startloc[cp]+iloc[cp]]=dst[i]-ifirst;
	    iloc[cp]++;
	    if (cp == myp){
		// I have this data
		double * s = a[cl]+c1;
		double sc = scale[dst[i]-ifirst];
		//		fprintf(stderr, "places to copy: %d %d scale=%e c1=%d,c2=%d\n",
		//			startloc[myp]+ii, dst[i]-ifirst, sc, c1,c2);
		//		fprintf(stderr,"s[0]=%e\n", s[0]);
		//		fprintf(stderr,"dst[0]=%e\n", *ar2[startloc[myp]+ii]);
		scalesubvect(sc,s,ar2[startloc[myp]+ii],c2-c1);
		//		fprintf(stderr, "end scalesubvect\n");
		ii++;
	    }
	}
    }
    //    fprintf(stderr,"loop end\n");
    //    dump_vector("bdst", bdst, nb);
//
//  now ar2 have (local share of) scaled umat
//  So do the Bcast  
    dumpsubmat("ar2 before bcast", c2-c1, ar2,  nb,  c2-c1, parms, controls);
    for(i=0;i<parms->nprow;i++){
	//	fprintf(stderr,"bcast proc %d lines %d\n", i, nlines[i]);
	if (nlines[i]){
	    MPI_Bcast(ar2[startloc[i]],(c2-c1)*nlines[i],MPI_DOUBLE, i,  controls->col_comm);
	}
    }
    dumpsubmat("ar2 after bcast", c2-c1, ar2,  nb,  c2-c1, parms, controls);
    
    //    dprintf(0,"global_swap, bcast part end\n");
    int nlinesd[parms->nprow];
    int startlocd[parms->nprow];
    int sdst[length];
    for(i=0;i<parms->nprow;i++)nlinesd[i]=0;
    for(i=0;i<parms->nprow;i++)iloc[i]=0;
    ii=0;
    for(i=0;i<length;i++){
	if(dst[i]>=ifirst+nb){
	    int cpd, cld;
	    convert_global_index_to_local_rows(dst[i], &cpd, &cld, parms, controls);
	    if(cpd == myp){
		sdst[ii]= cld;
		ii++;
	    }
	    nlinesd[cpd]++;
	}
    }
    //    dump_vector("nlinesd", nlinesd, parms->nprow);
    //    dump_vector("sdst", sdst, nlinesd[myp]);
    startlocd[0]=0;
    for(i=0;i<parms->nprow-1;i++)startlocd[i+1] = startlocd[i]+nlinesd[i];
    //
    //  now each processor knows where to store its data
    //
    if(current_proc == myp){
	//
	// this is swap data. All data comes from current_proc
	for(i=0;i<length;i++){
	    if(dst[i]>=ifirst+nb){
		int cpd, cld;
		int cps, cls;
		convert_global_index_to_local_rows(dst[i], &cpd, &cld, parms, controls);
		convert_global_index_to_local_rows(src[i], &cps, &cls, parms, controls);
		copysubvect(&(a[cls][c1]),arow[startlocd[cpd]+iloc[cpd]],c2-c1);
		iloc[cpd]++;
	    }
	}
	//
	// now arow (in current_proc) has all data to swap out
	// So send them out
	for(i=0;i<parms->nprow;i++){
	    if ((i != current_proc) && (nlinesd[i])){
		MPI_Send(arow[startlocd[i]], nlinesd[i]*(c2-c1), MPI_DOUBLE, i,
			 MPSENDBASETAG+i,controls->col_comm);
	    }
	}
    }else{
	//
	// other processors: receive data
	//
	MPI_Status status;
	if (nlinesd[myp]){
	    MPI_Recv(arow[startlocd[myp]], nlinesd[myp]*(c2-c1), MPI_DOUBLE, current_proc,
		     MPSENDBASETAG+myp,controls->col_comm, &status);
	}
    }
    //
    // now all data to be stored are in arow and ar2
    // first store arow (swapdata) to real locations
    //    dprintf(0,"p2p swap end\n");
    
    for(i=0;i<nlinesd[myp]; i++){
	copysubvect(arow[startlocd[myp]+i],&(a[sdst[i]][c1]), c2-c1);
    }
    //    dprintf(0,"p2p local swap end\n");

    // then store arow2 (UMAT) to arow
    for(i=0;i<nb; i++){
	//	dprintf(0,"bdst[%d]=%d\n", i, bdst[i]);
	copysubvect(ar2[i],arow[bdst[i]], c2-c1);
    }
    //    dprintf(0,"umat local copy end\n");
    // and copy arow back to a in the case of current_proc
    
    if(current_proc == myp){
	copysubmat(c2-c1, arow, nncol, (double(*)[])(a[current_lloc]+c1),  nb, c2-c1);
    }
    //    dprintf(0,"umat local copyback on current row end\n");
}



void process_right_part_mpi_using_dls_phase1(int ifirst,int nb,int c1, int c2,
					     int nnrow, int nncol,
					     double a[nnrow][nncol],
					     double acolinv[nb][nb],
					     double acol[nnrow][nb],
					     double arow[nb][c2-c1],
					     int pv[],
					     int src[],
					     int dst[],
					     int length,
					     double scale[],
					     PCONTROLS controls,  PPARMS parms)
{

    dump_vector("pv", pv, nb);
    //    fprintf(stderr,"Enter dls_phase1\n");
    if (parms->vcommscheme == 0){
	//	fprintf(stderr,"Enter global_swap_using...\n");
	global_swap_using_src_and_dest(ifirst, nb, c1, c2, nnrow, nncol, a,
				       arow, (double(*)[])(arow[nb]),
				       scale, src, dst, length,
				       controls,parms);
	return;
    }
    
    int i,ii, j;
    // arow can be used as work area before bcast_umat
    //
    //    dprintf(9,"process_rp_using_dls_p1, ifirst=%d c1, c2=%d %d\n", ifirst,c1,c2);
#if 1
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	//	fprintf(stderr,"dls_phase 1 ii=%d\n",ii);
	//	dprintf(0,"pv[%2d]=%2d\n", ii, pv[ii]);
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	MP_scale_row_blocked_using_scale(nnrow, nncol, a, parms,
					 controls,i,c1,c2,0,scale[ii]);
    }
#else
    
    local_swap_using_src_and_dest(ifirst, nb, c1, c2, nnrow, nncol, a, arow, scale, 
				  src,  dst, length, controls, parms);
#endif    
    //    print_current_time("end swap and scale");
    MP_bcast_umat(nnrow, nncol, a, parms, controls,ifirst,nb,c1,c2,
		      ifirst+nb,arow);
    //    print_current_time("end bcast_umat");
}

void process_right_part_mpi_using_dls_phase1_old(int ifirst,int nb,int c1, int c2,
					     int nnrow, int nncol,
					     double a[nnrow][nncol],
					     double acolinv[nb][nb],
					     double acol[nnrow][nb],
					     double arow[nb][c2-c1],
					     int pv[],
					     double scale[],
					     PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    // arow can be used as work area before bcast_umat
    //
    //    dprintf(9,"process_rp_using_dls_p1, ifirst=%d c1, c2=%d %d\n", ifirst,c1,c2);
    for (ii=0;ii<nb;ii++){
	i = ii+ifirst;
	//	dprintf(9,"process_rp, i=%d\n", i);
	MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
			     c1,c2);
	//	dprintf(9,"process_rp swap rows end, i=%d\n", i);
	MP_scale_row_blocked_using_scale(nnrow, nncol, a, parms,
					 controls,i,c1,c2,0,scale[ii]);
    }
    //    print_current_time("end swap and scale");
    MP_bcast_umat(nnrow, nncol, a, parms, controls,ifirst,nb,c1,c2,
		       ifirst+nb,arow);
    //    dprintf(9,"end bcast_umat\n");
}
void process_right_part_mpi_using_dls_phase2(int ifirst,int nb,int c1, int c2,
					     int nnrow, int nncol,
					     double a[nnrow][nncol],
					     double acolinv[nb][nb],
					     double acol[nnrow][nb],
					     double arow[nb][c2-c1],
					     int pv[],
					     double scale[],
					     PCONTROLS controls,  PPARMS parms)
{
    int i,ii;
    //    dprintf(9,"process_rp_using_dls_p2, ifirst=%d c1, c2=%d %d\n", ifirst,c1,c2);
    //    MP_update_multiple_blocked_global_using_lmat(nnrow, nncol, a, parms,
    //						 controls,ifirst,nb,c1,c2,
    //						 ifirst+nb,acol);
#if 1
    //    MP_bcast_umat(nnrow, nncol, a, parms, controls,ifirst,nb,c1,c2,
    //		       ifirst+nb,acol,arow);

    gdrsetforceswapab();
    MP_update_using_lu(nnrow, nncol, a, parms, controls,ifirst,nb,c1,c2,
		       ifirst+nb,acol,arow);
    gdrresetforceswapab();
#endif    
}

void column_decomposition_recursive_mpi_old(int ifirst,int nb,int nnrow, int nncol,
					double a[nnrow][nncol],
					double b[],  int pv[],
					PCONTROLS controls,  PPARMS parms)
{
    int  i,ii;

    dprintf(9,"column recursive %d %d\n", ifirst, nb);
    int havec=have_current_col(ifirst,  parms, controls);
    if (!havec) return;
    if (nb <= 8){
	dprintf(9,"column recursive calling transpose %d %d\n", ifirst, nb);
	column_decomposition_mpi_with_transpose(ifirst, nb, nnrow, nncol,
						a, b, pv,controls,parms);
	dprintf(9,"column recursive return from transpose %d %d\n", ifirst, nb);
    }else{	
	dprintf(9,"column recursive left part %d %d\n", ifirst, nb/2);
	column_decomposition_recursive_mpi_old(ifirst, nb/2, nnrow, nncol,
				 a, b, pv,controls,parms);
	dprintf(9,"column process right part %d %d\n", ifirst, nb/2);
	process_right_part_mpi(ifirst, nb/2, nb/2, nnrow, nncol, a, b,
			       pv,controls, parms,1);
	dprintf(9,"column recursive right part %d %d\n", ifirst+nb/2, nb/2);
	column_decomposition_recursive_mpi_old(ifirst+nb/2, nb/2, nnrow, nncol,
				 a, b, pv+nb/2,controls,parms);
	// process the swap of rows for the left half
	int c1, c2;
	convert_global_col_range_to_local_range(ifirst, ifirst+nb/2,
						&c1, &c2,  parms, controls);
	dprintf(9,"column recursive left part swap %d %d\n", ifirst, nb/2);
	for (ii=nb/2;ii<nb;ii++){
	    i = ii+ifirst;
	    MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
				 c1,c2);
	    MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,1);
	}
	dprintf(9,"column recursive left part swap end %d %d\n", ifirst, nb/2);
    }
}

void column_decomposition_recursive_mpi(int ifirst,int nb,int nnrow, int nncol,
					double a[nnrow][nncol],
					double (*acol)[],
					double b[],  int pv[],
					PCONTROLS controls,  PPARMS parms)
{
    int  i,ii;

    int havec=have_current_col(ifirst,  parms, controls);
    if (!havec) return;
    //    print_current_time("enter column_deconp");
    //    dprintf(9,"enter column_decomp, i, nb=%d %d\n", ifirst,nb);
    if (nb <= 8){
	BEGIN_TIMER(timer);
	column_decomposition_mpi_with_transpose(ifirst, nb, nnrow, nncol,
						a, b, pv,controls,parms);
	END_TIMER(timer,7,((double)(nb))*(nnrow-ifirst));
	
    }else{	
	column_decomposition_recursive_mpi(ifirst, nb/2, nnrow, nncol,
					   a, acol, b, pv,controls,parms);
	BEGIN_TIMER(timer0);
	process_right_part_mpi_withacol(ifirst, nb/2, nb/2, nnrow, nncol, a,
					acol, b,  pv,controls, parms,1);
	END_TIMER(timer0,8,((double)(nb/2))*(nb/2)*(nnrow-ifirst));
	column_decomposition_recursive_mpi(ifirst+nb/2, nb/2, nnrow, nncol,
					   a, acol, b, pv+nb/2,controls,parms);
	// process the swap of rows for the left half
	int c1, c2;
	convert_global_col_range_to_local_range(ifirst, ifirst+nb/2,
						&c1, &c2,  parms, controls);
	BEGIN_TIMER(timer1);
	for (ii=nb/2;ii<nb;ii++){
	    i = ii+ifirst;
	    MP_swap_rows_blocked(nnrow, nncol, a, parms, controls,i,pv[ii],
				 c1,c2);
	    MP_scale_row_blocked(nnrow, nncol, a, parms, controls,i,c1,c2,1);
	}
	END_TIMER(timer1,9,((double)(nb/2))*(nb/2));
    }
    //    print_current_time("end column_deconp");
    //    dprintf(9,"end column_decomp, i, nb=%d %d\n", ifirst,nb);
}



void lu_mpi(int nnrow, int nncol, double a[nnrow][nncol], double b[],
	    PCONTROLS controls,  PPARMS parms)
{
    int i, j, k, n;
    n = controls->nrow*parms->nprow;
    for(i=0;i<n;i++){
	lu_forward_onestep_mpi(i, nnrow, nncol, a, b, controls, parms);
    }
    backward_sub_mpi(nnrow, nncol, a, b,controls, parms);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
}


void lu_mpi_blocked(int nnrow, int nncol, double a[nnrow][nncol], double b[],
	    PCONTROLS controls,  PPARMS parms)
{
    int i, j, k, n, nb;
    n = controls->nrow*parms->nprow;
    nb = parms->nb;
    int pv[nb];
    double acolinv[nb][nb];
    double acol[nnrow][nb];
    for(i=0;i<n;i+=nb){
	//	dprintf(9,"Enter lu_mpi_blocked, i = %d\n",i);
	column_decomposition_recursive_mpi_old(i, nb, nnrow, nncol, a, b,
					   pv,controls, parms);
	MP_process_diagonal(nnrow, nncol, a, parms, controls,
			    i,nb,acolinv,0);

	MP_process_lmat(nnrow, nncol, a, parms, controls,i,nb,i+nb,acol);

	int c1, c2;
	convert_global_col_range_to_local_range(i+nb, i+nb-1,&c1, &c2,
						parms, controls);
	process_right_part_mpi_using_dl(i, nb, c1, nncol, nnrow, nncol, a,
					acolinv, acol, pv,controls, parms);
	//	process_right_part_mpi_using_dl_old(i, nb, 0, nnrow, nncol, a, acolinv,
	//			acol, pv,controls, parms);
    }
    backward_sub_mpi(nnrow, nncol, a, b,controls, parms);
    //    printmat_MP(nnrow, nncol, a, controls, parms);
}

int local_check_and_continue_rcomm_transfer();

void process_right_part_mpi_using_dls_concurrent(int i,int nb,int cfirst,
						 int nnrow, int nncol,
						 double a[nnrow][nncol],
						 double aip[nb][nb],
						 double acp[nnrow][nb],
						 double * arow,
						 double * arow2,
						 int * pvp,
						 double * scalep,
						 PCONTROLS controls,
						 PPARMS parms)
{
    int cs,ce,cenext;
    double (*arp)[];
    double (*arp2)[];
    arp= (double(*)[]) arow;
    arp2= (double(*)[]) arow2;
    int ninc;
    //    ninc= nb*4;
    //    if (ninc > 4096) ninc = nb*2;
    ninc= nb;
    // should not make bigger than 8 (arow memory limit might be exceeded)
    //    ninc= nb/2;
#ifndef CONCURRENT_UCOMM
    // non-overlapping comm
    //    ninc=nncol;
#endif
    BEGIN_TIMER(timerx);

    int n=parms->n;
    int src[n];
    int dst[n];
    int length;
    dumpsubmat("a before phase1", nncol, a, controls->nrow, controls->ncol, parms, controls);

    generate_src_and_dst_lists(i, pvp, src, dst, &length,controls, parms);

    
    for(cs=cfirst; cs<nncol;cs+= ninc){
	ce = cs + ninc;
	if (ce >= controls->ncol)ce = nncol;
	if (ce >nncol) ce = nncol;
	if (ce != nncol){
	    cenext = ce+ninc;
	    if (cenext >= controls->ncol)cenext = nncol;
	}
	if (cs == cfirst){
	    //	    dprintf(9,"using_dls call first phase1\n");
	    BEGIN_TIMER(timer);
	    print_current_time("enter first dls_phase_1");
	    process_right_part_mpi_using_dls_phase1(i, nb, cs, ce, nnrow,
						    nncol, a, aip, acp,arp,
						    pvp,src, dst, length, scalep,
						    controls, parms);
	    print_current_time("end first dls_phase_1");
	    dumpsubmat("a after phase1", nncol, a, controls->nrow, controls->ncol, parms, controls);
	    dumpsubmat("arow", ce-cs, arp, nb, ce-cs, parms, controls);
	    
	    END_TIMER(timer,3,((double)(ce-cs))*nb);
	    //	    print_current_time("end first dls_phase_1");
	    //	    dprintf(9,"using_dls end first phase1\n");
	}
#ifndef DO_RCOMM_LAST	
	if (controls->check_rcomm_transfer){
	    //	    dprintf(9,"using_dls call local_check_and_continue_rcomm\n");
	    int rcomm_state = local_check_and_continue_rcomm_transfer();
	    //	  dprintf(9,"using_dls, cs=%d,  rcomm=%d\n", cs, rcomm_state);
	}
	
#endif
#ifdef CONCURRENT_UCOMM
	omp_set_nested(1);
#pragma omp parallel
#pragma omp sections
#endif	
	{
#ifdef CONCURRENT_UCOMM		
#pragma omp section
#endif	
	    {
		print_current_time("enter dls_phase_2");
		BEGIN_TIMER(timer);
		process_right_part_mpi_using_dls_phase2(i, nb, cs, ce, nnrow,
							nncol, a, aip, acp,arp,
							pvp,scalep,
							controls, parms);
		gdrsetskipsendjmat();
		double x=parms->n - i;
		END_TIMER(timer,2,x*nb*(ce-cs));
		print_current_time("end dls_phase_2");
		
	    }
#ifdef CONCURRENT_UCOMM		
#pragma omp section
#endif	
	    if (ce < nncol){
#ifdef CONCURRENT_UCOMM		
                usleep(10000);
#endif	
		print_current_time("enter dls_phase_1");
		BEGIN_TIMER(timer);
		process_right_part_mpi_using_dls_phase1(i, nb, ce, cenext,
							nnrow, nncol,
							a, aip, acp,arp2,
							pvp, src, dst, length, scalep,
							controls, parms);
		END_TIMER(timer,3,((double)(ce-cs)*nb));
		print_current_time("end dls_phase_1");
	    }
	}
	
	swapvoidptr((void**)&arp2, (void**)&arp);
	if (ce == nncol) cs = nncol;
    }
    gdrresetskipsendjmat();
    double x=parms->n - i;
    END_TIMER(timerx,1,x*(x-nb)*nb*2);
}


void MP_process_row_comm_using_Bcast(int nnrow, int nncol,
				     double a[nnrow][nncol],
				     PCONTROLS controls,  PPARMS parms,
				     int i, int nb, int n, int * pvp2,
				     double * scalep2,
				     double (*aip2)[], double (*acp2)[])
{
    //    print_current_time("enter process_row_comm");
    if (i+nb < n){
	int current_prow;
	int current_lrow;
	
	int ii = i + nb;
	convert_global_index_to_local_rows(ii, &current_prow, &current_lrow,
					   parms, controls);
	int pcol = pcolid(ii,parms,controls);
	if (current_prow == controls->rank_in_col){
	    MPI_Bcast(aip2,sizeof(double)*nb*nb,MPI_BYTE,
		      pcol,  controls->row_comm);
	}
	int nrows=MP_prepare_lmat(nnrow, nncol, a, parms, controls,
				  ii,nb,ii+nb,acp2);
	MPI_Bcast(pvp2,sizeof(int)*nb,MPI_BYTE,
		  pcol,  controls->row_comm);
	MPI_Bcast(scalep2,sizeof(double)*nb,MPI_BYTE,
		  pcol, controls->row_comm);
	MPI_Bcast(acp2,sizeof(double)*nrows*nb,MPI_BYTE,
		  pcol, controls->row_comm);
    }
    print_current_time("end process_row_comm");
}
void MP_process_row_comm(int nnrow, int nncol, double a[nnrow][nncol],
			 PCONTROLS controls,  PPARMS parms,
			 int i, int nb, int n, int * pvp2, double * scalep2,
			 double (*aip2)[], double (*acp2)[])
{
    print_current_time("enter process_row_comm");
    if (i+nb < n){
	MPI_Status status; 
	int current_prow;
	int current_lrow;
	
	int ii = i + nb;
	BEGIN_TIMER(timer);
	convert_global_index_to_local_rows(ii, &current_prow, &current_lrow,
					   parms, controls);
	int nextp= (controls->rank_in_row+1)%parms->npcol;
	int prevp= (controls->rank_in_row-1+parms->npcol)%parms->npcol;
	int pcol = pcolid(ii,parms,controls);
	if (current_prow == controls->rank_in_col){
#if 0
	    if (controls->rank_in_row != pcol){
		MPI_Recv(aip2, sizeof(double)*nb*nb,MPI_BYTE,
			 prevp,MPRCOMMTAG, controls->row_comm, &status);
	    }
	    if (nextp != pcol){
		MPI_Send(aip2, sizeof(double)*nb*nb,MPI_BYTE,
			 nextp,MPRCOMMTAG, controls->row_comm);
	    }
#endif
	    //	    dprintf(9,"call bcast aip2\n");
	    MPI_Bcast(aip2,sizeof(double)*nb*nb,MPI_BYTE,
		      pcol,  controls->row_comm);
	}
	//	dprintf(9,"call prepare lmat\n");
	int nrows=MP_prepare_lmat(nnrow, nncol, a, parms, controls,
				  ii,nb,ii+nb,acp2);

	//	dprintf(9,"call bcast pvp2\n");
	MPI_Bcast(pvp2,sizeof(int)*nb,MPI_BYTE,
		  pcol,  controls->row_comm);
	//	dprintf(9,"call bcast scalp2\n");
	MPI_Bcast(scalep2,sizeof(double)*nb,MPI_BYTE,
		  pcol, controls->row_comm);
	//	dprintf(9,"call bcast acp2\n");
	//	MPI_Bcast(acp2,sizeof(double)*nrows*nb,MPI_BYTE,
	//		  pcol, controls->row_comm);
	MP_mybcast(acp2,sizeof(double)*nrows*nb, pcol, controls->row_comm);
	END_TIMER(timer,0,(double)((n-i-nb)*nb));
    }
    print_current_time("end process_row_comm");
}

static RCOMMT rcomm;

#define MAXPENDINGMESSAGE 100

void register_singlemessage_to_rcomm(PRCOMMT rcomm, void * p, int length)
{
    int i = rcomm->nmessages;
    (rcomm->message+i)->mptr=p;
    (rcomm->message+i)->length=length;
    (rcomm->message+i)->message_state = INITIALIZED;
    if (rcomm->first){
	(rcomm->message+i)->message_state = RECEIVED;
	if (rcomm->last){
	    (rcomm->message+i)->message_state = SENT;
	}
    }
    rcomm->nmessages++;
}

int check_and_continue_rcomm_transfer(PRCOMMT rcomm, int blockmode)
{
    int i;
    int nend = 0;
    MPI_Status status;
    int received_count = 0;
    int sent_count = 0;
    for(i=0;i<rcomm->nmessages; i++){
	PMESSAGE mp = rcomm->message+i;
	if ((mp->message_state == INITIALIZED)
	    && (i<MAXPENDINGMESSAGE+received_count)){
	    //	    fprintf(stderr,"new Irecv at %d\n",i);
	    int retval=MPI_Irecv(mp->mptr, mp->length, MPI_BYTE,rcomm->prevp,
				 MPNRMESSAGETAG+i,rcomm->comm, &(mp->request));
	    if(retval != MPI_SUCCESS)MP_error("MPI_Irecv error in start_rcomm");
	    mp->message_state = RECEIVING;
	}
	if (mp->message_state == RECEIVING){
	    int flag;
	    if (blockmode){
		MPI_Wait(&(mp->request), &status);
		flag = 1;
	    }else{
		MPI_Test(&(mp->request),&flag, &status);
	    }
	    if (flag)mp->message_state = RECEIVED;
	}
	if (mp->message_state == RECEIVED){
	    if(received_count < i)received_count = i;

	    if (rcomm->last){
		mp->message_state = SENT;
	    }else{
		if (i < sent_count + MAXPENDINGMESSAGE){
		    //		    fprintf(stderr,"new Isend at %d\n",i);
		    int retval=MPI_Isend(mp->mptr, mp->length, MPI_BYTE,rcomm->nextp,
					 MPNRMESSAGETAG+i,rcomm->comm, &(mp->request));
		    if(retval != MPI_SUCCESS)MP_error("MPI_Isend error in check_rcomm");
		    mp->message_state = SENDING;
		}
	    }
	}
	if (mp->message_state == SENDING){
	    int flag;
	    if (blockmode){
		MPI_Wait(&(mp->request), &status);
		flag = 1;
	    }else{
		MPI_Test(&(mp->request),&flag, &status);
	    }
	    if (flag)mp->message_state = SENT;
	}
	if (mp->message_state == SENDING) nend ++;
	if (mp->message_state == SENT){
	    if (sent_count < i)sent_count = i;
	}
    }
    return rcomm->nmessages - nend;
}

int local_check_and_continue_rcomm_transfer()
{
    return check_and_continue_rcomm_transfer(&rcomm, 0);
}


void start_rcomm_transfer(PRCOMMT rcomm)
{
    int i;
    //    dprintf(9,"Enter start_rcomm\n");
    for(i=0;(i<rcomm->nmessages) &&(i<MAXPENDINGMESSAGE); i++){
	PMESSAGE mp = rcomm->message+i;
	if (mp->message_state == INITIALIZED){
	    //	    dprintf(9,"Irecv register i=%d length=%d\n",i, mp->length);
	    int retval=MPI_Irecv(mp->mptr, mp->length, MPI_BYTE,rcomm->prevp,
				 MPNRMESSAGETAG+i,rcomm->comm, &(mp->request));
	    if(retval != MPI_SUCCESS)MP_error("MPI_Irecv error in start_rcomm");
	    mp->message_state = RECEIVING;
	}
	//	dprintf(9,"start_rcom, i=%d status=%d\n", i, mp->message_state);
    }
    check_and_continue_rcomm_transfer(rcomm,0);
}


void register_to_rcomm(PRCOMMT rcomm, void * p, int length)
{
    int i;
    for (i=0;i<length; i+= MAXRCOMMMESSAGE){
	int len  = MAXRCOMMMESSAGE;
	if (i+len > length) len = length-i;
	register_singlemessage_to_rcomm(rcomm, ((char*)p)+i, len);
    }
    
    
}


int MP_process_row_comm_init(int nnrow, int nncol, double a[nnrow][nncol],
			 PCONTROLS controls,  PPARMS parms,
			 int i, int nb, int n, int * pvp2, double * scalep2,
			 double (*aip2)[], double (*acp2)[])
{
    print_current_time("enter process_row_comm_init");
    controls->check_rcomm_transfer = 1;
    int nrows=0;
    if (i+nb < n){
	int current_prow;
	int current_lrow;
	
	int ii = i + nb;
	BEGIN_TIMER(timer);
	convert_global_index_to_local_rows(ii, &current_prow, &current_lrow,
					   parms, controls);
	rcomm.nextp= (controls->rank_in_row+1)%parms->npcol;
	rcomm.prevp= (controls->rank_in_row-1+parms->npcol)%parms->npcol;
	rcomm.nmessages=0;
	rcomm.first  = (controls->rank_in_row == pcolid(ii,parms,controls));
	rcomm.last = (rcomm.nextp == pcolid(ii,parms,controls));
	rcomm.comm = controls->row_comm;
	rcomm.preceive=0;
	rcomm.psend=0;
	if (current_prow == controls->rank_in_col){
	    register_to_rcomm(&rcomm, aip2, nb*nb*sizeof(double));
	}
	nrows=MP_prepare_lmat(nnrow, nncol, a, parms, controls,
				  ii,nb,ii+nb,acp2);
	register_to_rcomm(&rcomm, pvp2, sizeof(int)*nb);
	register_to_rcomm(&rcomm, scalep2,sizeof(double)*nb);
	register_to_rcomm(&rcomm, acp2,sizeof(double)*nrows*nb);
	start_rcomm_transfer(&rcomm);
    }
    print_current_time("end process_row_comm_init");
    return nrows;
}

void MP_process_row_comm_test(int nnrow, int nncol, double a[nnrow][nncol],
			 PCONTROLS controls,  PPARMS parms,
			 int i, int nb, int n, int * pvp2, double * scalep2,
			 double (*aip2)[], double (*acp2)[])
{
    MP_process_row_comm_init(nnrow, nncol, a, controls, parms, i, nb, n,
			     pvp2, scalep2, aip2, acp2);
    check_and_continue_rcomm_transfer(&rcomm,1);
    //    MPI_Barrier(controls->row_comm);
}



void lu_mpi_blocked_lookahead(int nnrow, int nncol, double a[nnrow][nncol],
			      double (*acol)[],double (*acol2)[],
			      double (*dinv)[],
			      double arow[],double arow2[],
			      double b[], PCONTROLS controls,  PPARMS parms)
{
    int i, j, k, n, nb;
    n = controls->nrow*parms->nprow;
    nb = parms->nb;
    int pv[nb];
    int pv2[nb];
    double scale[nb];
    double scale2[nb];
    double acolinv[nb][nb] __attribute__((aligned(128)));
    double acolinv2[nb][nb] __attribute__((aligned(128)));
    double (*aip)[];
    double (*aip2)[];
    double (*acp)[];
    double (*acp2)[];
    double (*arp)[];
    double (*arp2)[];
    double (*aptmp)[];
    int * pvp;
    int * pvp2;
    int * pvptmp;
    double * scalep;
    double  * scalep2;
    double * scaleptmp;
    i=0;
    print_current_time("enter first rfact");
    BEGIN_TIMER(timertotal);
    BEGIN_TIMER(timerx);
    BEGIN_TIMER(timercd);
    //    void gdrsetusemultithread();
    column_decomposition_recursive_mpi(i, nb, nnrow, nncol, a, acol, b,
				       pv,controls, parms);
    END_TIMER(timercd,30,(double)((n-i-nb)*nb));
    BEGIN_TIMER(timer00);
    print_current_time("end first rfact");
    //    dprintf(9,"call bcast pv\n");
    MPI_Bcast(pv,sizeof(int)*nb,MPI_BYTE, pcolid(i,parms,controls),
	      controls->row_comm);
    MP_construct_scalevector(nnrow, nncol,  a, parms, controls, i, nb, scale);

    MPI_Bcast(scale,sizeof(double)*nb,MPI_BYTE, pcolid(i,parms,controls),
	      controls->row_comm);
    print_current_time("enter process_diagonal");
    MP_process_diagonal(nnrow, nncol, a, parms, controls,
			i,nb,acolinv,0);
    print_current_time("end MP_process_diagonal");
    int nrows=MP_process_lmat(nnrow, nncol, a, parms, controls,i,nb,i+nb,acol);
    print_current_time("end MP_process_lmat");
    MP_calculate_ld(nb, acol,  nrows, acol2, acolinv,i,controls, parms);
    END_TIMER(timerx,5,(double)((n-i-nb)*nb));
    END_TIMER(timer00,37,0.0);
    END_TIMER(timer00,39,0.0);
    print_current_time("end first lookahead");

    aip=acolinv;
    acp = acol;
    aip2=acolinv2;
    acp2 = acol2;
    pvp=pv;
    pvp2=pv2;
    scalep=scale;
    scalep2=scale2;
    for(i=0;i<n;i+=nb){
	//	void gdrsetusemultithread();
	//	fprintf(stderr,"lu2_mpi i=%d\n",i);
	int c1, c2, cfirst;
	int havec = have_current_col(i+nb,  parms, controls);
	convert_global_col_range_to_local_range(i+nb, i+nb*2-1,&c1, &c2,
						parms, controls);
	arp= (double(*)[]) arow;
	c2 ++;
	BEGIN_TIMER(timerx);
	print_current_time("enter rfact");

	int src[n];
	int dst[n];
	int length;
	dump_vector("pvp", pvp, nb);
	generate_src_and_dst_lists(i, pvp, src, dst, &length,controls, parms);

	if ((i+nb < n) && havec){
	    //	if (0){
	    int ii = i + nb;
	    BEGIN_TIMER(timer00);

	    //	    dprintf(9,"rfact call dls_phase1, ii, nb=%d %d\n", ii, nb);
	    dumpsubmat("a before first phase1", nncol, a, controls->nrow,
		       controls->ncol, parms, controls);
	    process_right_part_mpi_using_dls_phase1(i, nb, c1, c2, nnrow,
						    nncol, a, aip, acp,arp,
						    pvp,src, dst, length, scalep,
						    controls, parms);
	    dumpsubmat("a after first phase1", nncol, a, controls->nrow, controls->ncol,
		       parms, controls);
	    //	    dprintf(9,"rfact call dls_phase2, ii, nb=%d %d\n", ii, nb);
	    process_right_part_mpi_using_dls_phase2(i, nb, c1, c2, nnrow,
						    nncol, a, aip, acp,arp,
						    pvp,scalep,
						    controls, parms);
	    //	    dprintf(9,"rfact call column_dec ii, nb=%d %d\n", ii, nb);
	    END_TIMER(timer00,37,0.0);
	    END_TIMER(timer00,40,0.0);
	    BEGIN_TIMER(timercd);
	    
	    column_decomposition_recursive_mpi(ii, nb, nnrow, nncol, a, acp2,
					       b, pvp2,controls, parms);
	    END_TIMER(timercd,30,(double)((n-i-nb)*nb));
	    //	    dprintf(9,"rfact call MP_construct_scale ii, nb=%d %d\n", ii, nb);
	    MP_construct_scalevector(nnrow, nncol,  a, parms,
				     controls, ii, nb, scalep2);
	    //	    dprintf(9,"rfact end, ii, nb=%d %d\n", ii, nb);
	    cfirst=c2;
	}else{
	    cfirst=c1;
	}
	BEGIN_TIMER(timer01);
	if (i+nb < n){
	    int ii = i + nb;
	    //	    dprintf(9,"call phase1 %d\n",i);
	    MP_process_diagonal_phase1(nnrow, nncol, a, parms, controls,
				ii,nb,aip2,0);
	    //	    dprintf(9,"call phase1 end\n");
	}
	END_TIMER(timer01,37,0.0);
	END_TIMER(timer01,41,0.0);
	END_TIMER(timerx,5,(double)((n-i-nb)*nb));
	print_current_time("end rfact");

	//	dprintf(9,"call process_row_comm\n");
	BEGIN_TIMER(timer03);
#ifndef DO_RCOMM_LAST	
	nrows=MP_process_row_comm_init(nnrow, nncol,  a, controls, parms,
				 i, nb, n, pvp2, scalep2,aip2, acp2);
#endif	
	//	dprintf(9,"call process_right_part_... %d\n",i);
	//	check_and_continue_rcomm_transfer(&rcomm,1);
	// test to finish rcomm here -- 2011/6/19
	process_right_part_mpi_using_dls_concurrent(i, nb, cfirst,
						    nnrow, nncol,
						    a, aip, acp, arow,
						    arow2,
						    pvp, scalep,
						    controls, parms);
	BEGIN_TIMER(timery);
	BEGIN_TIMER(timer02);
	//	dprintf(9,"end process_right_part_... %d\n",i);
#ifdef DO_RCOMM_LAST	
	nrows=MP_process_row_comm_init(nnrow, nncol,  a, controls, parms,
				 i, nb, n, pvp2, scalep2,aip2, acp2);
#endif	
	check_and_continue_rcomm_transfer(&rcomm,1);
	//	dprintf(9,"end check_and_continue... %d\n",i);
	
	print_current_time("end lmat/dls");	
#ifndef DUPOSTMUL	
	MP_update_multiple_using_diagonal(nnrow, nncol, a, parms,
					  controls,
					  i,c1,nncol,nb,aip);
#else	
	MP_update_multiple_using_diagonal(nnrow, nncol, a, parms,
					  controls,
					  i,controls->ncol,controls->ncol+1,nb,aip);
#endif	

	MP_store_diagonal_inverse(nnrow, nb, dinv, parms,controls, i, aip);

	print_current_time("end mult_diag");
	
	if (i+nb < n){
	    int ii = i + nb;
	    MP_calculate_ld_phase1(nb, acp2, nrows, aip2, ii,
				   controls, parms);
	    print_current_time("end ld_phase1");
	    MP_calculate_ld_phase2(nb, acp2, nrows, acp, aip2, ii,controls, parms);
	    print_current_time("end ld_phase2");
	}
	//	fprintf(stderr, "before swap\n");
	dump_vector("pvp2", pvp2, nb);
	dump_vector("pvp ", pvp,  nb);
	swapvoidptr((void**)&acp2, (void**)&acp);
	swapvoidptr((void**)&aip2, (void**)&aip);
	swapvoidptr((void**)&pvp2, (void**)&pvp);
	swapvoidptr((void**)&scalep2, (void**)&scalep);
	//	fprintf(stderr, "after swap\n");
	dump_vector("pvp2", pvp2, nb);
	dump_vector("pvp ", pvp,  nb);
	if (MP_myprocid()==0)  fprintf(stderr,"lu_mpi look i=%d end\n", i);
	END_TIMER(timer02,37,0.0);
	END_TIMER(timer02,42,0.0);
	END_TIMER(timery,5,0.0);
    }
    END_TIMER(timertotal,4,((double)(n))*((double)n)*(n*2.0/3.0));
    print_current_time("enter backward_sub");
    BEGIN_TIMER(timerback);
    backward_sub_blocked_mpi(nnrow, nncol, a,nb, dinv, b,controls, parms);
    END_TIMER(timerback,28,(double)(n)*n);
    print_current_time("end backward_sub");
    //    printmat_MP(nnrow, nncol, a, controls, parms);
}

    

void usage()
{
    fprintf(stderr,"lu2_mpi options:\n");
    fprintf(stderr,"  -h: This help\n");
    fprintf(stderr,"  -s: seed (default=1)\n");
    fprintf(stderr,"  -n: size of matrix (default=8192)\n");
    fprintf(stderr,"  -r: allocate processors in row-major order \n");
    fprintf(stderr,"  -b: block size (default=2048)\n");
    fprintf(stderr,"  -p: processors in row (default=1)\n");
    fprintf(stderr,"  -q: processors in column (default=1)\n");
    fprintf(stderr,"  -g: usehugetlbfs (default=no)\n");
    fprintf(stderr,"  -w: two process per node mode (use two cards)\n");
    fprintf(stderr,"  -B: first card ID (default=0)\n");
    fprintf(stderr,"  -N: Number of cards per MPI process(default=1)\n");
    fprintf(stderr,"  -T: Max process ID for timing info(default=0: all)\n");
    fprintf(stderr,"  -v: scheme for vertical communication (default=0: advanced, else: simple)\n");
    fprintf(stderr,"  -S: stress factor for thermal test(default=0: no additional stress)\n");
}


extern char *optarg;
extern int optind;

void print_parms(FILE* stream, PPARMS parms)
{
    fprintf(stream,"N=%d Seed=%d NB=%d\n", parms->n,parms->seed,parms->nb);
    fprintf(stream,"P=%d Q=%d Procs row major=%d usehuge=%d\n",
	    parms->nprow,parms->npcol,
	    parms->procs_row_major,
	    parms->usehugepage);
    fprintf(stream,"usetwocards=%d ncards=%d cardid=%d maxpt=%d vcommscheme=%d stress=%d\n",
	    parms->twocards,
	    parms->ncards,
	    parms->firstcard,
	    parms->maxpidfortiming,
	    parms->vcommscheme,
	    parms->stress_factor);
}

void read_parms(int argc, char * argv[], PPARMS parms)
{
    int ch;
    static struct option longopts[] = {
	{ "help",      no_argument,      0,           'h' },
	{ "block_size",      optional_argument,      NULL,           'b' },
	{ "seed",      optional_argument,            NULL,           's' },
	{ "ndim_matrix",   required_argument,      NULL,           'n' },
	{ "processors_in_row",  optional_argument,   NULL,     'p' },
	{ "processors_in_column",  optional_argument,   NULL,     'q' },
	{ "processors_row_major",  no_argument,   0,     'r' },
	{ "usehugepage",  no_argument,   0,     'g' },
	{ "usetwocards",  no_argument,   0,     'w' },
	{ "ncards",  optional_argument,   NULL,     'N' },
	{ "max_pid_for_timing",  optional_argument,   NULL,     'T' },
	{ "vcomm_scheme",  optional_argument,   NULL,     'v' },
	{ "stress_factor",  optional_argument,   NULL,     'S' },
	{ NULL,         0,                      NULL,           0 }
    };
    MP_message("enter read_parms");
    parms->seed=1;
    parms->n=8192;
    parms->nb = 2048;
    parms->nprow=1;
    parms->npcol=1;
    parms->procs_row_major =  0;
    parms->usehugepage =  0;
    parms->twocards =  0;
    parms->ncards =  1;
    parms->firstcard =  0;
    parms->maxpidfortiming =  0;
    parms->vcommscheme =  0;
    parms->stress_factor=0;
    for(argc=0;argv[argc]!=NULL;argc++);
    // the above is necessary to fixup the argc value messed up by MPICH...
    // JM 2009/9/21
    while((ch=getopt_long(argc,argv,"B:N:S:T:b:ghn:p:q:rs:v:w",longopts, NULL))!= -1){
	fprintf(stderr,"optchar = %c optarg=%s\n", ch,optarg);
	switch (ch) {
	    case 'B': parms->firstcard = atoi(optarg); break;
	    case 'N': parms->ncards = atoi(optarg); break;
	    case 'S': parms->stress_factor = atoi(optarg); break;
	    case 'T': parms->maxpidfortiming = atoi(optarg); break;
	    case 'b': parms->nb = atoi(optarg); break;
	    case 'g': parms->usehugepage = 1; break;
	    case 's': parms->seed = atoi(optarg); break;
	    case 'n': parms->n = atoi(optarg); break;
	    case 'p': parms->npcol = atoi(optarg); break;
	    case 'q': parms->nprow = atoi(optarg); break;
	    case 'v': parms->vcommscheme = atoi(optarg); break;
	    case 'r': parms->procs_row_major = 1; break;
	    case 'w': parms->twocards = 1; break;
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
	fprintf(stderr,"P, Q, MP_proccount = %d %d %d\n",
		parms->npcol, parms->nprow,MP_proccount());
    if (parms->nprow*parms->npcol != MP_proccount()){
	MP_error("P*Q != np");
    }
}

void MP_broadcast_parms(PPARMS parms)
{
    fprintf(stderr,"broadcast_parms size= %d\n", sizeof(PARMS));
    MP_bcast((void*)parms, sizeof(PARMS));
    MP_message("broadcast_parms normal end");
    print_parms(stderr, parms);
}

void set_parms(int argc, char * argv[], PPARMS parms)
{
    MP_message("Enter set_parms");
    if (MP_myprocid()==0)read_parms(argc, argv, parms);
    MP_message("read_parms end");
    MP_broadcast_parms(parms);
    MP_message("broadcast end");
}
    

void setup_controls(PPARMS parms, PCONTROLS controls)
{
    controls->nrow = parms->n/parms->nprow;
    controls->ncol = parms->n/parms->npcol;
    controls->nnrow = controls->nrow;
    controls->nncol = controls->ncol+NNCOLINC;
    controls->check_rcomm_transfer = 0;
}


int main(int argc, char * argv[])
{
    int ch;
    PARMS parms;
    CONTROLS controls;
    int i;

    MP_initialize(&argc,&argv);
    MP_message("Return from MP_initialize");
    set_parms(argc, argv, &parms);
    setup_controls(&parms, &controls);
    MP_message("Return from setup_controls");


    int nb, n, seed;
    nb = parms.nb;
    n = parms.n;
    seed = parms.seed;
    double (*a)[];
    double (*acol)[];
    double (*acol2)[];
    double (*dinv)[];
    double *arow, *arow2;
    int * pv;
    double *b, *b0, *bcopy;
    long int nl=n;

    set_max_pid_for_print_time(parms.maxpidfortiming);
    gdrsetboardid(parms.firstcard);
    if (parms.twocards){
	gdrsetboardid(MP_myprocid() %2);
    }

    gdrdgemm_set_stress_factor(parms.stress_factor);

    if (parms.maxpidfortiming){
	if (MP_myprocid() >= parms.maxpidfortiming)set_matmul_msg_level(0);
    }
    if(parms.ncards > 1)    gdrsetnboards(parms.ncards);
    if (parms.usehugepage){
	dprintf(9,"hugetlb: usehuge = %d\n", parms.usehugepage);
	MP_allocate_hugetlbfs("/mnt/huge/aaa", (void**)&a, (void**)&acol,
			      (void**)&acol2,(void**)&dinv,(void**)&arow,
			      (void**)&arow2, controls.nnrow,
			      nb,controls.nncol);
    }else{
	int nbb = nb+32;
	dprintf(9,"malloc: usehuge = %d\n", parms.usehugepage);
	a = (double(*)[]) malloc(sizeof(double)*controls.nnrow*controls.nncol);
	acol = (double(*)[]) malloc(sizeof(double)*controls.nnrow*nbb);
	acol2 = (double(*)[]) malloc(sizeof(double)*controls.nnrow*nbb);
	dinv = (double(*)[]) malloc(sizeof(double)*controls.nnrow*nbb);
	arow = (double*) malloc(sizeof(double)*nb*nbb*2);
	arow2 = (double*) malloc(sizeof(double)*nb*nbb*2);
    }
    fprintf(stderr,"Return from MP_allocate\n");

    //    init_matoutfid();

    if(MP_myprocid())set_debug_level(8);


    {
	//	omp_set_nested(1);
	//	fprintf(stderr,"Omp_get_nested=%d\n", omp_get_nested());
    }

    b = (double*)malloc(sizeof(double)*controls.nnrow);
    b0 = (double*)malloc(sizeof(double)*controls.nnrow);
    bcopy = (double*)malloc(sizeof(double)*parms.n);
    pv = (int*)malloc(sizeof(int)*controls.nnrow);

    MP_setup_communicators(&parms, &controls);
    fprintf(stderr,"Return from MP_setup_comm");

    sleep(MP_myprocid()%10);

    if (controls.nnrow >  nb*4){
	reset_gdr(nb, acol2, nb, acol, controls.nnrow);
    }else{
	reset_gdr(nb, arow2, nb, arow, nb*4);
    }
    fprintf(stderr,"Proc %d Return from reset_gdr\n", MP_myprocid());
    if (seed == 0){
	readmat(n,a);
    }else{
	MP_randomsetmat(controls.nncol, controls.nnrow, a,&parms,
			&controls,1,b0);
    }
#if 1
    MP_sync();
    if (controls.nnrow >  nb*4){
	reset_gdr(nb, acol2, nb, acol, controls.nnrow);
    }else{
	reset_gdr(nb, arow2, nb, arow, nb*4);
    }
#endif
    MP_sync();
    fprintf(stderr,"Proc %d Return from reset_gdr\n", MP_myprocid());
    timer_init();
    init_timer();
    if (MP_myprocid()==0){
	print_current_datetime("Start calculation ");
    }
    init_current_time();
    MP_message("enter lu_mpi");
    lu_mpi_blocked_lookahead(controls.nnrow, controls.nncol, a,
			     acol,acol2,dinv,arow,arow2,
			     b,&controls, &parms);
    //    dprintf(9,"end lu_mpi_lookahead\n");
    //    printmat_MP(controls.nnrow, controls.nncol, a, &controls, &parms);
    double ctime=cpusec();
    double wtime=wsec();
    if (MP_myprocid()==0){
	print_current_datetime("End calculation ");
    }
    if (seed == 0){
	readmat(n,a);
    }else{
	MP_randomsetmat(controls.nncol, controls.nnrow, a,&parms,
			&controls,0,b0);
    }
    dprintf(9,"end randomsetmat\n");
    double a1, ao, b1, bo;
    MP_calcnorm(controls.nncol, controls.nnrow, a, &parms, &controls, &a1, &ao);
    MP_calcvnorm(b, &parms, &controls, &b1, &bo);

    check_solution_mpi(controls.nnrow, controls.nncol, a, b, bcopy,
		       &controls, &parms);
    //    dprintf(9,"end check_solution\n");
    //    printmat_MP(controls.nnrow, controls.nncol, a, &controls, &parms);

    double einf = print_errors( b,b0,&controls, &parms);
    print_timers_mpi(0);
    if (MP_myprocid()==0){
	double nd = parms.n;
	HPLlogprint(stderr, parms.n, parms.nb, parms.nprow, parms.npcol, wtime);
	HPLerrorprint(stderr, einf,  a1, ao, b1, bo, nd);
	fflush(stderr);
	fflush(stdout);
	fprintf(stderr,"\n");
	fprintf(stderr,"\n");
	fprintf(stderr,"\n");
	fprintf(stderr,"\n");
	HPLlogprint(stdout, parms.n, parms.nb, parms.nprow, parms.npcol, wtime);
	HPLerrorprint(stdout, einf,  a1, ao, b1, bo, nd);
	//	printf("\n\n   Norms = %e %e %e %e\n",  a1, ao, b1, bo);
	double speed = nd*nd*nd*2.0/3.0/wtime/1e9;
	fprintf(stderr,"\n\n   cpusec =  %g wsec=%g %g Gflops\n\n\n",   ctime, wtime, speed);
	printf("\n\n   cpusec =  %g wsec=%g %g Gflops\n\n\n",   ctime, wtime, speed);
	fflush(stdout);
    }
    //    MP_message("ALL end --- perform MPI_Barrier\n");
    MPI_Barrier(MPI_COMM_WORLD);
    //    MP_message("MPI_Barrier end\n");
    MPI_Barrier(MPI_COMM_WORLD);
    //    MP_message("Second MPI_Barrier end\n");
    MPI_Finalize();
    //    MP_message("MPI_Finalize end\n");

    return 0;
}
