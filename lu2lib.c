//
// lu2lib.c
//
// J. Makino
//    Time-stamp: <11/06/20 17:50:32 makino>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <timerlib.h>
#ifndef NOBLAS
#ifdef MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#endif
#ifdef USEGDR
#include "gdrdgemm.h"
#endif

#define FTYPE double
#include <emmintrin.h>
typedef double v2df __attribute__((vector_size(16)));
typedef union {v2df v; double s[2];}v2u;

#ifndef USEGDR
void  gdrsetboardid(int boardid)
{}
#endif


void matmul2_host(int n,
		  FTYPE a[n][n],
		  FTYPE b[n][n],
		  FTYPE c[n][n])

{
    int i, j, k;
    for(i=0;i<n;i++){
	for(j=0;j<n;j++){
	    c[i][j]=0.0e0;
	    for(k=0;k<n;k++) c[i][j]+= a[i][k]*b[k][j];
	}
    }
}

// simplest version
void matmul_for_small_nk_0(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{
    // simplest version
    int i,j,k;
    for(j=0;j<n;j++)
	for(i=0;i<m;i++)
	    for(k=0;k<kk;k++)
		c[i][j] -= a[i][k]*b[k][j];
}

// make copy of B
void matmul_for_small_nk_1(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j,k;
    double bcopy[n][kk];
    for(k=0;k<kk;k++)
	for(j=0;j<n;j++)
		bcopy[j][k] = b[k][j];
    for(i=0;i<m;i++){
	for(j=0;j<n;j++){
	    register double tmp=0.0;
	    for(k=0;k<kk;k++){
		tmp += a[i][k]*bcopy[j][k];
	    }
	    c[i][j] -= tmp;
	}
    }
}
// hand-unroll innermost loop
void matmul_for_small_nk_2(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j,k;
    double bcopy[n][kk];
    for(j=0;j<n;j++)
	    for(k=0;k<kk;k++)
		bcopy[j][k] = b[k][j];
    for(i=0;i<m;i++){
	double *ap=a[i];
	for(j=0;j<n;j++){
	    double *bp = bcopy[j];
	    double tmp=0.0;
	    for(k=0;k<kk;k+=8)
		tmp += ap[k]*bp[k]
		    + ap[k+1]*bp[k+1]
		    + ap[k+2]*bp[k+2]
		    + ap[k+3]*bp[k+3]
		    + ap[k+4]*bp[k+4]
		    + ap[k+5]*bp[k+5]
		    + ap[k+6]*bp[k+6]
		    + ap[k+7]*bp[k+7];
	    c[i][j]-=tmp;
	}
    }
}
// hand-unroll mid-loop
void matmul_for_small_nk_3(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j,k;
    double bcopy[n][kk];
    for(j=0;j<n;j++)
	    for(k=0;k<kk;k++)
		bcopy[j][k] = b[k][j];
    for(i=0;i<m;i++){
	double *ap=a[i];
	for(j=0;j<n;j+=4){
	    double *bp = bcopy[j];
	    double *bpp = bcopy[j+1];
	    double *bp2 = bcopy[j+2];
	    double *bp3 = bcopy[j+3];
	    double tmp=0.0;
	    double tmp1=0.0;
	    double tmp2=0.0;
	    double tmp3=0.0;
	    for(k=0;k<kk;k+=8){
		tmp += ap[k]*bp[k]
		    + ap[k+1]*bp[k+1]
		    + ap[k+2]*bp[k+2]
		    + ap[k+3]*bp[k+3]
		    + ap[k+4]*bp[k+4]
		    + ap[k+5]*bp[k+5]
		    + ap[k+6]*bp[k+6]
		    + ap[k+7]*bp[k+7];
		tmp1 += ap[k]*bpp[k]
		    + ap[k+1]*bpp[k+1]
		    + ap[k+2]*bpp[k+2]
		    + ap[k+3]*bpp[k+3]
		    + ap[k+4]*bpp[k+4]
		    + ap[k+5]*bpp[k+5]
		    + ap[k+6]*bpp[k+6]
		    + ap[k+7]*bpp[k+7];
		tmp2 += ap[k]*bp2[k]
		    + ap[k+1]*bp2[k+1]
		    + ap[k+2]*bp2[k+2]
		    + ap[k+3]*bp2[k+3]
		    + ap[k+4]*bp2[k+4]
		    + ap[k+5]*bp2[k+5]
		    + ap[k+6]*bp2[k+6]
		    + ap[k+7]*bp2[k+7];
		tmp3 += ap[k]*bp3[k]
		    + ap[k+1]*bp3[k+1]
		    + ap[k+2]*bp3[k+2]
		    + ap[k+3]*bp3[k+3]
		    + ap[k+4]*bp3[k+4]
		    + ap[k+5]*bp3[k+5]
		    + ap[k+6]*bp3[k+6]
		    + ap[k+7]*bp3[k+7];
	    }
	    c[i][j]-=tmp;
	    c[i][j+1]-=tmp1;
	    c[i][j+2]-=tmp2;
	    c[i][j+3]-=tmp3;
	}
    }
}
// hand-unroll mid-loop by 2
void matmul_for_small_nk_4(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j,k;
    double bcopy[n][kk];
    for(j=0;j<n;j++)
	    for(k=0;k<kk;k++)
		bcopy[j][k] = b[k][j];
    for(i=0;i<m;i++){
	double *ap=a[i];
	for(j=0;j<n;j+=2){
	    double *bp = bcopy[j];
	    double *bpp = bcopy[j+1];
	    double tmp=0.0;
	    double tmp1=0.0;
	    for(k=0;k<kk;k+=8){
		tmp += ap[k]*bp[k]
		    + ap[k+1]*bp[k+1]
		    + ap[k+2]*bp[k+2]
		    + ap[k+3]*bp[k+3]
		    + ap[k+4]*bp[k+4]
		    + ap[k+5]*bp[k+5]
		    + ap[k+6]*bp[k+6]
		    + ap[k+7]*bp[k+7];
		tmp1 += ap[k]*bpp[k]
		    + ap[k+1]*bpp[k+1]
		    + ap[k+2]*bpp[k+2]
		    + ap[k+3]*bpp[k+3]
		    + ap[k+4]*bpp[k+4]
		    + ap[k+5]*bpp[k+5]
		    + ap[k+6]*bpp[k+6]
		    + ap[k+7]*bpp[k+7];
	    }
	    c[i][j]-=tmp;
	    c[i][j+1]-=tmp1;
	}
    }
}
// use sse2 for dot product
void matmul_for_small_nk_5(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j,k;
    int nh = n/2;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    double acopyd[kk];
    for(j=0;j<nh;j++)
	    for(k=0;k<kk;k++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	double *ap=a[i];
	double *acp = (double*) acopy;
	register v2df tmp= (v2df){0.0,0.0};
	v2df * cp = (v2df*) (&(c[i][0]));
	for(k=0;k<kk;k+=4){
	     __builtin_prefetch((double*)a[i+4]+k,0);
	}
	for(j=0;j<n;j+=4){
	     __builtin_prefetch(c[i+4]+j,0);
	}

	
	for(k=0;k<kk;k+=2){
	    //	    v2df aa = *((v2df*)(ap+k));
	    //	    acopy[k]=__builtin_ia32_shufpd(aa,aa,0x0);
	    //	    acopy[k+1]= __builtin_ia32_shufpd(aa,aa,0x5);
	    acp[k*2]=acp[k*2+1]=ap[k];
	    acp[k*2+2]=acp[k*2+3]=ap[k+1];
	}
	for(j=0;j<nh;j++){
	    tmp = (v2df){0.0,0.0};
	    v2df * bp = bcopy2[j];
	    for(k=0;k<kk;k+=2){
		tmp += acopy[k]*bp[k]
		    +acopy[k+1]*bp[k+1]
#if 0		    
		    +acopy[k+2]*bp[k+2]
		    +acopy[k+3]*bp[k+3]
		    +acopy[k+4]*bp[k+4]
		    +acopy[k+5]*bp[k+5]
		    +acopy[k+6]*bp[k+6]
		    +acopy[k+7]*bp[k+7]
#endif		    
		    ;

	    }
	    cp[j] -= tmp;
	}
    }
}
// use sse2 for dot product
void matmul_for_small_nk_6(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j,k;
    int nh = n/2;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    v2df acopy3[kk];
    v2df acopy4[kk];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    double *acp3 = (double*) acopy3;
    double *acp4 = (double*) acopy4;
    for(j=0;j<nh;j++)
	    for(k=0;k<kk;k++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i+=4){
	double *ap=a[i];
	double *ap2=a[i+1];
	double *ap3=a[i+2];
	double *ap4=a[i+3];
	register v2df tmp, tmp2, tmp3, tmp4;
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df * cp2 = (v2df*) (&(c[i+1][0]));
	v2df * cp3 = (v2df*) (&(c[i+2][0]));
	v2df * cp4 = (v2df*) (&(c[i+3][0]));
	for(k=0;k<kk;k+=4){
	     __builtin_prefetch((double*)a[i+4]+k,0);
	     __builtin_prefetch((double*)a[i+5]+k,0);
	     __builtin_prefetch((double*)a[i+6]+k,0);
	     __builtin_prefetch((double*)a[i+7]+k,0);
	     __builtin_prefetch(c[i+4]+j,0);
	     __builtin_prefetch(c[i+5]+j,0);
	     __builtin_prefetch(c[i+6]+j,0);
	     __builtin_prefetch(c[i+7]+j,0);
	}


	
	for(k=0;k<kk;k+=2){
	    v2df * aa = (v2df*)(ap+k);
	    acopy [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    aa = (v2df*)(ap2+k);
	    acopy2 [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy2[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    //   acp[k*2]=acp[k*2+1]=ap[k];
	    //	    acp[k*2+2]=acp[k*2+3]=ap[k+1];
	    //	    acp2[k*2]=acp2[k*2+1]=ap2[k];
	    //	    acp2[k*2+2]=acp2[k*2+3]=ap2[k+1];
	}
	for(k=0;k<kk;k+=2){
	    acp3[k*2]=acp3[k*2+1]=ap3[k];
	    acp3[k*2+2]=acp3[k*2+3]=ap3[k+1];
	    acp4[k*2]=acp4[k*2+1]=ap4[k];
	    acp4[k*2+2]=acp4[k*2+3]=ap4[k+1];
	}
	for(j=0;j<nh;j++){
	    tmp = tmp2= tmp3= tmp4= (v2df){0.0,0.0};
	    v2df * bp = bcopy2[j];
#if 0	    
	    for(k=0;k<kk;k+=4){
		tmp += acopy[k]*bp[k]
		    +acopy[k+1]*bp[k+1]
		    +acopy[k+2]*bp[k+2]
		    +acopy[k+3]*bp[k+3];
		tmp2 += acopy2[k]*bp[k]
		    +acopy2[k+1]*bp[k+1]
		    +acopy2[k+2]*bp[k+2]
		    +acopy2[k+3]*bp[k+3];
		tmp3 += acopy3[k]*bp[k]
		    +acopy3[k+1]*bp[k+1]
		    +acopy3[k+2]*bp[k+2]
		    +acopy3[k+3]*bp[k+3];
		tmp4 += acopy4[k]*bp[k]
		    +acopy4[k+1]*bp[k+1]
		    +acopy4[k+2]*bp[k+2]
		    +acopy4[k+3]*bp[k+3];

	    }
#endif	    
	    for(k=0;k<kk;k+=2){
		tmp += acopy[k]*bp[k]
		    +acopy[k+1]*bp[k+1];
		tmp2 += acopy2[k]*bp[k]
		    +acopy2[k+1]*bp[k+1];
		tmp3 += acopy3[k]*bp[k]
		    +acopy3[k+1]*bp[k+1];
		tmp4 += acopy4[k]*bp[k]
		    +acopy4[k+1]*bp[k+1];

	    }
	    cp[j] -= tmp;
	    cp2[j] -= tmp2;
	    cp3[j] -= tmp3;
	    cp4[j] -= tmp4;
	}
    }
}
void matmul_for_small_nk7(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    //    BEGIN_TSC;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    //    END_TSC(bpcount);
    for(i=0;i<m;i+=2){
	//	BEGIN_TSC;
	double *ap=a[i];
	double *ap2=a[i+1];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df * cp2 = (v2df*) (&(c[i+1][0]));

	
	for(k=0;k<kk;k+=2){
	    v2df * aa = (v2df*)(ap+k);
	     __builtin_prefetch((double*)a[i+4]+k,0);
	     __builtin_prefetch((double*)a[i+5]+k,0);
	    acopy [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    aa = (v2df*)(ap2+k);
	    acopy2 [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy2[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    //   acp[k*2]=acp[k*2+1]=ap[k];
	    //	    acp[k*2+2]=acp[k*2+3]=ap[k+1];
	    //	    acp2[k*2]=acp2[k*2+1]=ap2[k];
	    //	    acp2[k*2+2]=acp2[k*2+3]=ap2[k+1];
	}
	//	END_TSC(apcount);
	//	BEGIN_TSC;
	for(j=0;j<nh;j++){
	    tmp = tmp2=  (v2df){0.0,0.0};
	    v2df ctmp= cp[j];
	    v2df ctmp2 = cp2[j] ;

	    v2df * bp = bcopy2[j];
	    __builtin_prefetch(c[i+4]+j,0);
	    __builtin_prefetch(c[i+5]+j,0);
	    for(k=0;k<kk;k+=8){
		int k2 = k+4;
		v2df *avp = acopy+k;
		v2df *avp2 = acopy2+k;
		v2df *bvp = bp+k;
		tmp += avp[0]*bvp[0];
		tmp2 += avp2[0]*bvp[0];
		tmp +=avp[1]*bvp[1];
		tmp2+=avp2[1]*bvp[1];
		tmp +=avp[2]*bvp[2];
		tmp2+=avp2[2]*bvp[2];
		tmp +=avp[3]*bvp[3];
		tmp2+=avp2[3]*bvp[3];
		tmp += avp[4]*bvp[4];
		tmp2 += avp2[4]*bvp[4];
		tmp +=avp[5]*bvp[5];
		tmp2+=avp2[5]*bvp[5];
		tmp +=avp[6]*bvp[6];
		tmp2+=avp2[6]*bvp[6];
		tmp +=avp[7]*bvp[7];
		tmp2+=avp2[7]*bvp[7];
	    }
	    
#if 0	    
	    for(k=0;k<kk;k+=8){
		int k2 = k+4;
		tmp += acopy[k]*bp[k];
		tmp2 += acopy2[k]*bp[k];
		tmp +=acopy[k+1]*bp[k+1];
		tmp2+=acopy2[k+1]*bp[k+1];
		tmp +=acopy[k+2]*bp[k+2];
		tmp2+=acopy2[k+2]*bp[k+2];
		tmp +=acopy[k+3]*bp[k+3];
		tmp2+=acopy2[k+3]*bp[k+3];
		tmp += acopy[k2]*bp[k2];
		tmp2 += acopy2[k2]*bp[k2];
		tmp +=acopy[k2+1]*bp[k2+1];
		tmp2+=acopy2[k2+1]*bp[k2+1];
		tmp +=acopy[k2+2]*bp[k2+2];
		tmp2+=acopy2[k2+2]*bp[k2+2];
		tmp +=acopy[k2+3]*bp[k2+3];
		tmp2+=acopy2[k2+3]*bp[k2+3];
	    }
#endif	    
#if 0
	    for(k=0;k<kk;k+=2){
		tmp += acopy[k]*bp[k];
		tmp 	    __builtin_prefetch(c[i+4+(j&1)]+j,0);
+=acopy[k+1]*bp[k+1];
		tmp2 += acopy2[k]*bp[k];
		tmp2+=acopy2[k+1]*bp[k+1];
	    }
#endif	    
	    cp[j] = ctmp -tmp;
	    cp2[j] = ctmp2 -tmp2;

	}
	//	END_TSC(dotcount);

    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
    
}


// XMM registers
#define X0	"%xmm0"
#define X1	"%xmm1"
#define X2	"%xmm2"
#define X3	"%xmm3"
#define X4	"%xmm4"
#define X5	"%xmm5"
#define X6	"%xmm6"
#define X7	"%xmm7"
#define X8		"%xmm8"
#define X9		"%xmm9"
#define X10		"%xmm10"
#define X11		"%xmm11"
#define X12		"%xmm12"
#define X13		"%xmm13"
#define X14		"%xmm14"
#define X15		"%xmm15"

#define LOADPD(mem, reg) asm("movapd %0, %"reg::"m"(mem));
#define STORPD(reg, mem) asm("movapd %"reg " , %0"::"m"(mem));
#define MOVNTPD(reg, mem) asm("movntpd %"reg " , %0"::"m"(mem));
#define MOVAPD(src, dst) asm("movapd " src "," dst);
#define MOVQ(src, dst) asm("movq " src "," dst);
#define BCAST0(reg) asm("shufpd $0x00, " reg ","  reg);
#define BCAST1(reg) asm("shufpd $0xff, " reg ","  reg);
#define MULPD(src, dst) asm("mulpd " src "," dst);
#define ADDPD(src, dst) asm("addpd " src ","  dst);
#define SUBPD(src, dst) asm("subpd "  src "," dst);


void matmul_for_nk8_0(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 8;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
#define PREFETCHL 32    
    for(i=0;i<PREFETCHL;i++){
	__builtin_prefetch((double*)a[i],0,0);
	__builtin_prefetch(c[i+8],1,0);
    }
    
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	//	v2df acopy[8];
	v2df *ap = (v2df*) a[i];
	v2df * cp = (v2df*) (&(c[i][0]));
	
	__builtin_prefetch((double*)a[i+PREFETCHL],0,0);

	int k;
	
	for(j=0;j<nh;j+=4){
	    __builtin_prefetch(c[i+PREFETCHL]+j,1,3);
	    v2df * bvp0 = bcopy2[j];
	    v2df * bvp1 = bcopy2[j+1];
	    v2df * bvp2 = bcopy2[j+2];
	    v2df * bvp3 = bcopy2[j+3];
	    LOADPD(cp[j],X12);
	    LOADPD(cp[j+1],X13);
	    LOADPD(cp[j+2],X14);
	    LOADPD(cp[j+3],X15);

	    LOADPD(ap[0],X0);
	    LOADPD(bvp0[0],X4);
	    LOADPD(bvp1[0],X5);
	    LOADPD(bvp2[0],X6);
	    LOADPD(bvp3[0],X7);
	    LOADPD(bvp0[1],X8);
	    LOADPD(bvp1[1],X9);
	    LOADPD(bvp2[1],X10);
	    LOADPD(bvp3[1],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);

	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);
	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[1],X0);

	    LOADPD(bvp0[2],X4);
	    LOADPD(bvp1[2],X5);
	    LOADPD(bvp2[2],X6);
	    LOADPD(bvp3[2],X7);
	    LOADPD(bvp0[3],X8);
	    LOADPD(bvp1[3],X9);
	    LOADPD(bvp2[3],X10);
	    LOADPD(bvp3[3],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);
	    
	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);
	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[2],X0);
	    LOADPD(bvp0[4],X4);
	    LOADPD(bvp1[4],X5);
	    LOADPD(bvp2[4],X6);
	    LOADPD(bvp3[4],X7);
	    LOADPD(bvp0[5],X8);
	    LOADPD(bvp1[5],X9);
	    LOADPD(bvp2[5],X10);
	    LOADPD(bvp3[5],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);

	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);
	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[3],X0);
	    
	    LOADPD(bvp0[6],X4);
	    LOADPD(bvp1[6],X5);
	    LOADPD(bvp2[6],X6);
	    LOADPD(bvp3[6],X7);
	    LOADPD(bvp0[7],X8);
	    LOADPD(bvp1[7],X9);
	    LOADPD(bvp2[7],X10);
	    LOADPD(bvp3[7],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);
	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);
	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);
	    STORPD(X12,cp[j+0]);
	    STORPD(X13,cp[j+1]);
	    STORPD(X14,cp[j+2]);
	    STORPD(X15,cp[j+3]);
	}
    }
}


void matmul_for_nk16_0a(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 16;
    int nh = n/2;
    register int k;
    v2df bcopy2[nh][kk];
#ifdef     PREFETCHL
#undef PREFETCHL
#endif
#define PREFETCHL 32
    
    for(i=0;i<PREFETCHL;i++){
	__builtin_prefetch((double*)a[i],0,0);
	__builtin_prefetch((double*)a[i]+8,0,0);
	__builtin_prefetch(c[i+8],1,0);
	__builtin_prefetch(c[i+8]+8,1,0);
    }
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i+=2){
	//	BEGIN_TSC;
	v2df *ap = (v2df*) a[i];
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df *app = (v2df*) a[i+1];
	v2df * cpp = (v2df*) (&(c[i+1][0]));
	
	__builtin_prefetch((double*)a[i+PREFETCHL],0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL]+8,0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL+1],0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL+1]+8,0,0);

	int k;
	
	for(j=0;j<nh;j+=2){
	    __builtin_prefetch(c[i+PREFETCHL]+j,1,0);
	    v2df * bvp0 = bcopy2[j];
	    v2df * bvp1 = bcopy2[j+1];
	    LOADPD(cp[j],X12);
	    LOADPD(cp[j+1],X13);
	    LOADPD(cpp[j],X14);
	    LOADPD(cpp[j+1],X15);

	    for(k=0;k<8;k++){
		
		LOADPD(ap[k],X0);
		LOADPD(app[k],X2);
		MOVAPD(X0,X1);
		BCAST0(X0);
		BCAST1(X1);
		MOVAPD(X2,X3);
		BCAST0(X2);
		BCAST1(X3);
		LOADPD(bvp0[k*2],X4);
		MOVAPD(X4,X6);
		MULPD(X0,X4);
		SUBPD(X4,X12);
		LOADPD(bvp1[k*2],X5);
		MOVAPD(X5,X7);
		MULPD(X0,X5);
		SUBPD(X5,X13);
		LOADPD(bvp0[k*2+1],X8);
		MOVAPD(X8,X10);
		MULPD(X1,X8);
		SUBPD(X8,X12);
		LOADPD(bvp1[k*2+1],X9);
		MOVAPD(X9,X11);
		MULPD(X1,X9);
		SUBPD(X9,X13);
		MULPD(X2,X6);
		SUBPD(X6,X14);
		MULPD(X2,X7);
		SUBPD(X7,X15);
		MULPD(X3,X10);
		SUBPD(X10,X14);
		MULPD(X3,X11);
		SUBPD(X11,X15);
	    }

	    STORPD(X12,cp[j+0]);
	    STORPD(X13,cp[j+1]);
	    STORPD(X14,cpp[j+0]);
	    STORPD(X15,cpp[j+1]);
	}
    }
}
void matmul_for_nk16_0c(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 16;
    int nh = n/2;
    register int k;
    v2df bcopy2[nh][kk];
#ifdef     PREFETCHL
#undef PREFETCHL
#endif
#define PREFETCHL 16
    
    for(i=0;i<PREFETCHL;i++){
	__builtin_prefetch((double*)a[i],0,0);
	__builtin_prefetch((double*)a[i]+8,0,0);
	__builtin_prefetch(c[i+8],1,0);
	__builtin_prefetch(c[i+8]+8,1,0);
    }
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	v2df *ap = (v2df*) a[i];
	v2df * cp = (v2df*) (&(c[i][0]));
	
	__builtin_prefetch((double*)a[i+PREFETCHL],0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL]+8,0,0);

	int k;
	
	for(j=0;j<nh;j+=4){
	    __builtin_prefetch(c[i+PREFETCHL]+j,1,0);
	    v2df * bvp0 = bcopy2[j];
	    v2df * bvp1 = bcopy2[j+1];
	    v2df * bvp2 = bcopy2[j+2];
	    v2df * bvp3 = bcopy2[j+3];
	    LOADPD(cp[j],X12);
	    LOADPD(cp[j+1],X13);
	    LOADPD(cp[j+2],X14);
	    LOADPD(cp[j+3],X15);

	    for(k=0;k<8;k++){
		
		LOADPD(ap[k],X0);
		LOADPD(bvp0[k*2],X4);
		LOADPD(bvp1[k*2],X5);
		LOADPD(bvp2[k*2],X6);
		LOADPD(bvp3[k*2],X7);
		MOVAPD(X0,X1);
		BCAST0(X0);
		BCAST1(X1);
		LOADPD(bvp0[k*2+1],X8);
		LOADPD(bvp1[k*2+1],X9);
		LOADPD(bvp2[k*2+1],X10);
		LOADPD(bvp3[k*2+1],X11);
		MULPD(X0,X4);
		MULPD(X0,X5);
		MULPD(X0,X6);
		MULPD(X0,X7);
		MULPD(X1,X8);
		MULPD(X1,X9);
		MULPD(X1,X10);
		MULPD(X1,X11);
		SUBPD(X4,X12);
		SUBPD(X5,X13);
		SUBPD(X6,X14);
		SUBPD(X7,X15);
		SUBPD(X8,X12);
		SUBPD(X9,X13);
		SUBPD(X10,X14);
		SUBPD(X11,X15);
	    }

	    STORPD(X12,cp[j+0]);
	    STORPD(X13,cp[j+1]);
	    STORPD(X14,cp[j+2]);
	    STORPD(X15,cp[j+3]);
	}
    }
}
void matmul_for_nk32_0(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 32;
    int nh = n/2;
    register int k;
    v2df bcopy2[nh][kk];
#ifdef     PREFETCHL
#undef PREFETCHL
#endif
#define PREFETCHL 8
    
    for(i=0;i<PREFETCHL;i++){
	__builtin_prefetch((double*)a[i],0,0);
	__builtin_prefetch((double*)a[i]+8,0,0);
	__builtin_prefetch((double*)a[i]+16,0,0);
	__builtin_prefetch((double*)a[i]+24,0,0);
	__builtin_prefetch(c[i+8],1,0);
	__builtin_prefetch(c[i+8]+8,1,0);
	__builtin_prefetch(c[i+8]+16,1,0);
	__builtin_prefetch(c[i+8]+24,1,0);
    }
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	v2df *ap = (v2df*) a[i];
	v2df * cp = (v2df*) (&(c[i][0]));
	
	__builtin_prefetch((double*)a[i+PREFETCHL],0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL]+8,0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL]+16,0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL]+24,0,0);

	int k;
	
	for(j=0;j<nh;j+=4){
	    __builtin_prefetch(c[i+PREFETCHL]+j,1,0);
	    v2df * bvp0 = bcopy2[j];
	    v2df * bvp1 = bcopy2[j+1];
	    v2df * bvp2 = bcopy2[j+2];
	    v2df * bvp3 = bcopy2[j+3];
	    LOADPD(cp[j],X12);
	    LOADPD(cp[j+1],X13);
	    LOADPD(cp[j+2],X14);
	    LOADPD(cp[j+3],X15);

	    for(k=0;k<16;k++){
		
		LOADPD(ap[k],X0);
		LOADPD(bvp0[k*2],X4);
		LOADPD(bvp1[k*2],X5);
		LOADPD(bvp2[k*2],X6);
		LOADPD(bvp3[k*2],X7);
		MOVAPD(X0,X1);
		BCAST0(X0);
		BCAST1(X1);
		MULPD(X0,X4);
		MULPD(X0,X5);
		MULPD(X0,X6);
		MULPD(X0,X7);
		LOADPD(bvp0[k*2+1],X8);
		LOADPD(bvp1[k*2+1],X9);
		LOADPD(bvp2[k*2+1],X10);
		LOADPD(bvp3[k*2+1],X11);
		MULPD(X1,X8);
		MULPD(X1,X9);
		MULPD(X1,X10);
		MULPD(X1,X11);
		SUBPD(X4,X12);
		SUBPD(X5,X13);
		SUBPD(X6,X14);
		SUBPD(X7,X15);
		SUBPD(X8,X12);
		SUBPD(X9,X13);
		SUBPD(X10,X14);
		SUBPD(X11,X15);
	    }

	    STORPD(X12,cp[j+0]);
	    STORPD(X13,cp[j+1]);
	    STORPD(X14,cp[j+2]);
	    STORPD(X15,cp[j+3]);
	}
    }
}

void matmul_for_nk16_0b(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 16;
    int nh = n/2;
    register int k;
    v2df bcopy2[nh][kk];
#ifdef     PREFETCHL
#undef PREFETCHL
#endif
#define PREFETCHL 16
    
    for(i=0;i<PREFETCHL;i++){
	__builtin_prefetch((double*)a[i],0,0);
	__builtin_prefetch((double*)a[i]+8,0,0);
	__builtin_prefetch(c[i+8],1,0);
	__builtin_prefetch(c[i+8]+8,1,0);
    }
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	v2df *ap = (v2df*) a[i];
	v2df * cp = (v2df*) (&(c[i][0]));
	
	__builtin_prefetch((double*)a[i+PREFETCHL],0,0);
	__builtin_prefetch((double*)a[i+PREFETCHL]+8,0,0);

	int k;
	
	for(j=0;j<nh;j+=4){
	    __builtin_prefetch(c[i+PREFETCHL]+j,1,0);
	    v2df * bvp0 = bcopy2[j];
	    v2df * bvp1 = bcopy2[j+1];
	    v2df * bvp2 = bcopy2[j+2];
	    v2df * bvp3 = bcopy2[j+3];
	    LOADPD(cp[j],X12);
	    LOADPD(cp[j+1],X13);
	    LOADPD(cp[j+2],X14);
	    LOADPD(cp[j+3],X15);

	    LOADPD(ap[0],X0);
	    LOADPD(bvp0[0],X4);
	    LOADPD(bvp1[0],X5);
	    LOADPD(bvp2[0],X6);
	    LOADPD(bvp3[0],X7);
	    LOADPD(bvp0[1],X8);
	    LOADPD(bvp1[1],X9);
	    LOADPD(bvp2[1],X10);
	    LOADPD(bvp3[1],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);

	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[1],X0);
	    
	    LOADPD(bvp0[2],X4);
	    LOADPD(bvp1[2],X5);
	    LOADPD(bvp2[2],X6);
	    LOADPD(bvp3[2],X7);
	    LOADPD(bvp0[3],X8);
	    LOADPD(bvp1[3],X9);
	    LOADPD(bvp2[3],X10);
	    LOADPD(bvp3[3],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);
	    
	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[2],X0);
	    LOADPD(bvp0[4],X4);
	    LOADPD(bvp1[4],X5);
	    LOADPD(bvp2[4],X6);
	    LOADPD(bvp3[4],X7);
	    LOADPD(bvp0[5],X8);
	    LOADPD(bvp1[5],X9);
	    LOADPD(bvp2[5],X10);
	    LOADPD(bvp3[5],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);
	    
	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[3],X0);
	    LOADPD(bvp0[6],X4);
	    LOADPD(bvp1[6],X5);
	    LOADPD(bvp2[6],X6);
	    LOADPD(bvp3[6],X7);
	    LOADPD(bvp0[7],X8);
	    LOADPD(bvp1[7],X9);
	    LOADPD(bvp2[7],X10);
	    LOADPD(bvp3[7],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);
	    
	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);

	    LOADPD(ap[4],X0);

	    LOADPD(bvp0[8],X4);
	    LOADPD(bvp1[8],X5);
	    LOADPD(bvp2[8],X6);
	    LOADPD(bvp3[8],X7);
	    LOADPD(bvp0[9],X8);
	    LOADPD(bvp1[9],X9);
	    LOADPD(bvp2[9],X10);
	    LOADPD(bvp3[9],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);
	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);
	    
	    LOADPD(ap[5],X0);
	    LOADPD(bvp0[10],X4);
	    LOADPD(bvp1[10],X5);
	    LOADPD(bvp2[10],X6);
	    LOADPD(bvp3[10],X7);
	    LOADPD(bvp0[11],X8);
	    LOADPD(bvp1[11],X9);
	    LOADPD(bvp2[11],X10);
	    LOADPD(bvp3[11],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);

	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);
	    
	    LOADPD(ap[6],X0);
	    LOADPD(bvp0[12],X4);
	    LOADPD(bvp1[12],X5);
	    LOADPD(bvp2[12],X6);
	    LOADPD(bvp3[12],X7);
	    LOADPD(bvp0[13],X8);
	    LOADPD(bvp1[13],X9);
	    LOADPD(bvp2[13],X10);
	    LOADPD(bvp3[13],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);

	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);
	    
	    LOADPD(ap[7],X0);
	    LOADPD(bvp0[14],X4);
	    LOADPD(bvp1[14],X5);
	    LOADPD(bvp2[14],X6);
	    LOADPD(bvp3[14],X7);
	    LOADPD(bvp0[15],X8);
	    LOADPD(bvp1[15],X9);
	    LOADPD(bvp2[15],X10);
	    LOADPD(bvp3[15],X11);
	    MOVAPD(X0,X1);
	    BCAST0(X0);
	    BCAST1(X1);

	    MULPD(X0,X4);
	    MULPD(X0,X5);
	    MULPD(X0,X6);
	    MULPD(X0,X7);
	    MULPD(X1,X8);
	    MULPD(X1,X9);
	    MULPD(X1,X10);
	    MULPD(X1,X11);
	    SUBPD(X4,X12);
	    SUBPD(X5,X13);
	    SUBPD(X6,X14);
	    SUBPD(X7,X15);

	    SUBPD(X8,X12);
	    SUBPD(X9,X13);
	    SUBPD(X10,X14);
	    SUBPD(X11,X15);
	    
	    STORPD(X12,cp[j+0]);
	    STORPD(X13,cp[j+1]);
	    STORPD(X14,cp[j+2]);
	    STORPD(X15,cp[j+3]);
	}
    }
}


void matmul_for_nk8_0d(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 8;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    v2df awork[4];
    v2df awork2[4];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	double *ap=a[i];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	
	v2df * aa = (v2df*)(ap);
	__builtin_prefetch((double*)a[i+8],0,0);

	v2df acopy0=(v2df){a[i][0], a[i][0]};
	v2df acopy1=(v2df){a[i][1], a[i][1]};
	v2df acopy2=(v2df){a[i][2], a[i][2]};
	v2df acopy3=(v2df){a[i][3], a[i][3]};
	v2df acopy4=(v2df){a[i][4], a[i][4]};
	v2df acopy5=(v2df){a[i][5], a[i][5]};
	v2df acopy6=(v2df){a[i][6], a[i][6]};
	v2df acopy7=(v2df){a[i][7], a[i][7]};
	v2df zero=(v2df){0.0, 0.0};

	LOADPD(acopy0,X0);
	LOADPD(acopy1,X1);
	LOADPD(acopy2,X2);
	LOADPD(acopy3,X3);
	LOADPD(acopy4,X4);
	LOADPD(acopy5,X5);
	LOADPD(acopy6,X6);
	LOADPD(acopy7,X7);
	
	for(j=0;j<nh;j++){
	    __builtin_prefetch(c[i+8]+j,1,0);
	    v2df * bvp = bcopy2[j];
	    LOADPD(cp[j],X14);
	    LOADPD(bvp[0],X8);
	    LOADPD(bvp[1],X9);
	    MULPD(X0,X8);
	    MULPD(X1,X9);
	    LOADPD(bvp[2],X10);
	    LOADPD(bvp[3],X11);
	    ADDPD(X9,X8);
	    MULPD(X2,X10);
	    MULPD(X3,X11);
	    ADDPD(X11,X10);
	    LOADPD(bvp[4],X9);
	    LOADPD(bvp[5],X11);
	    LOADPD(bvp[6],X12);
	    LOADPD(bvp[7],X13);
	    MULPD(X4,X9);
	    MULPD(X5,X11);
	    ADDPD(X10,X8);
	    ADDPD(X11,X9);
	    MULPD(X6,X12);
	    MULPD(X7,X13);
	    ADDPD(X13,X12);
	    ADDPD(X9,X8);
	    ADDPD(X12,X8);
	    SUBPD(X8,X14);
	    STORPD(X14,cp[j]);
	}
    }
}



void matmul_for_nk8_0c(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 8;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    v2df awork[4];
    v2df awork2[4];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	double *ap=a[i];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	
	v2df * aa = (v2df*)(ap);
	__builtin_prefetch((double*)a[i+8],0,0);

	register v2df acopy0=(v2df){a[i][0], a[i][0]};
	register v2df acopy1=(v2df){a[i][1], a[i][1]};
	register v2df acopy2=(v2df){a[i][2], a[i][2]};
	register v2df acopy3=(v2df){a[i][3], a[i][3]};
	register v2df acopy4=(v2df){a[i][4], a[i][4]};
	register v2df acopy5=(v2df){a[i][5], a[i][5]};
	register v2df acopy6=(v2df){a[i][6], a[i][6]};
	register v2df acopy7=(v2df){a[i][7], a[i][7]};
	
	for(j=0;j<nh;j++){
	    tmp =  (v2df){0.0,0.0};
	    v2df ctmp= cp[j];

	    v2df * bp = bcopy2[j];
	    __builtin_prefetch(c[i+4]+j,1,0);
	    v2df *bvp = bp;
	    tmp += acopy0*bvp[0];
	    tmp +=acopy1*bvp[1];
	    tmp +=acopy2*bvp[2];
	    tmp +=acopy3*bvp[3];
	    tmp +=acopy4*bvp[4];
	    tmp +=acopy5*bvp[5];
	    tmp +=acopy6*bvp[6];
	    tmp +=acopy7*bvp[7];
	    cp[j] = ctmp -tmp;
	}
    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
    
}
void matmul_for_nk8_0b(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 8;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    v2df awork[4];
    v2df awork2[4];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i++){
	//	BEGIN_TSC;
	double *ap=a[i];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	
	v2df * aa = (v2df*)(ap);
	__builtin_prefetch((double*)a[i+8],0,0);

	acopy[0]=(v2df){a[i][0], a[i][0]};
	acopy[1]=(v2df){a[i][1], a[i][1]};
	acopy[2]=(v2df){a[i][2], a[i][2]};
	acopy[3]=(v2df){a[i][3], a[i][3]};
	acopy[4]=(v2df){a[i][4], a[i][4]};
	acopy[5]=(v2df){a[i][5], a[i][5]};
	acopy[6]=(v2df){a[i][6], a[i][6]};
	acopy[7]=(v2df){a[i][7], a[i][7]};
	
	for(j=0;j<nh;j++){
	    tmp = tmp2=  (v2df){0.0,0.0};
	    v2df ctmp= cp[j];

	    v2df * bp = bcopy2[j];
	    __builtin_prefetch(c[i+4]+j,1,0);
	    v2df *avp = acopy;
	    v2df *bvp = bp;
	    tmp += avp[0]*bvp[0];
	    tmp +=avp[1]*bvp[1];
	    tmp +=avp[2]*bvp[2];
	    tmp +=avp[3]*bvp[3];
	    tmp += avp[4]*bvp[4];
	    tmp +=avp[5]*bvp[5];
	    tmp +=avp[6]*bvp[6];
	    tmp +=avp[7]*bvp[7];
	    cp[j] = ctmp -tmp;
	}
    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
    
}
void matmul_for_nk8_0a(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j;
    int kk = 8;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    v2df awork[4];
    v2df awork2[4];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i+=2){
	//	BEGIN_TSC;
	double *ap=a[i];
	double *ap2=a[i+1];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df * cp2 = (v2df*) (&(c[i+1][0]));

	
	v2df * aa = (v2df*)(ap);
	__builtin_prefetch((double*)a[i+8],0,0);
	__builtin_prefetch((double*)a[i+9],0,0);

	acopy[0]=(v2df){a[i][0], a[i][0]};
	acopy[1]=(v2df){a[i][1], a[i][1]};
	acopy[2]=(v2df){a[i][2], a[i][2]};
	acopy[3]=(v2df){a[i][3], a[i][3]};
	acopy[4]=(v2df){a[i][4], a[i][4]};
	acopy[5]=(v2df){a[i][5], a[i][5]};
	acopy[6]=(v2df){a[i][6], a[i][6]};
	acopy[7]=(v2df){a[i][7], a[i][7]};
	
	aa = (v2df*)(ap2);
	acopy2[0]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	acopy2[1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	aa++;
	acopy2[2]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	acopy2[3]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	aa++;
	acopy2[4]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	acopy2[5]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	aa++;
	acopy2[6]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	acopy2[7]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	aa++;
	for(j=0;j<nh;j++){
	    tmp = tmp2=  (v2df){0.0,0.0};
	    v2df ctmp= cp[j];
	    v2df ctmp2 = cp2[j] ;

	    v2df * bp = bcopy2[j];
	    __builtin_prefetch(c[i+4]+j,1,0);
	    __builtin_prefetch(c[i+5]+j,1,0);
	    v2df *avp = acopy;
	    v2df *avp2 = acopy2;
	    v2df *bvp = bp;
	    tmp += avp[0]*bvp[0];
	    tmp2 += avp2[0]*bvp[0];
	    tmp +=avp[1]*bvp[1];
	    tmp2+=avp2[1]*bvp[1];
	    tmp +=avp[2]*bvp[2];
	    tmp2+=avp2[2]*bvp[2];
	    tmp +=avp[3]*bvp[3];
	    tmp2+=avp2[3]*bvp[3];
	    tmp += avp[4]*bvp[4];
	    tmp2 += avp2[4]*bvp[4];
	    tmp +=avp[5]*bvp[5];
	    tmp2+=avp2[5]*bvp[5];
	    tmp +=avp[6]*bvp[6];
	    tmp2+=avp2[6]*bvp[6];
	    tmp +=avp[7]*bvp[7];
	    tmp2+=avp2[7]*bvp[7];
	    cp[j] = ctmp -tmp;
	    cp2[j] = ctmp2 -tmp2;

	}

    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
    
}
void matmul_for_nk8_1(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j,k;
    const int kk = 8;
    const int kh = kk/2;
    int nh = n/2;
    v2df bcopy[n][kh];
    v2df acopy[kk][kh];
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    for(j=0;j<n;j++)
	for(k=0;k<kh;k++)
	    bcopy[j][k] = (v2df){b[k*2][j],b[k*2+1][j]};
    //    printf("copy b end\n");
    for(i=0;i<m;i+=kk){
	for(k=0;k<kk;k++){
	    v2df  *ak = (v2df*)(a[i+k]);
	    v2df * awp =acopy+k;
	    awp[0]=ak[0];
	    awp[1]=ak[1];
	    awp[2]=ak[2];
	    awp[3]=ak[3];
	}
	//	printf("copy a end\n");
	for(k=0;k<kk;k++){
	    v2u tmp, tmp1;
	    v2df * ap = acopy[k];
	    for(j=0;j<n;j+=2){
		tmp.v = ap[0]*bcopy[j][0]
		    + ap[1]*bcopy[j][1]
		    + ap[2]*bcopy[j][2]
		    + ap[3]*bcopy[j][3];
		tmp1.v = ap[0]*bcopy[j+1][0]
		    + ap[1]*bcopy[j+1][1]
		    + ap[2]*bcopy[j+1][2]
		    + ap[3]*bcopy[j+1][3];
		c[k+i][j] -= tmp.s[0]+tmp.s[1];
		c[k+i][j+1] -= tmp1.s[0]+tmp1.s[1];
	    }
	}
	//	printf("calc c end\n");
    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
}
void matmul_for_nk8_2(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j,k;
    const int kk = 8;
    const int kh = kk/2;
    int nh = n/2;
    v2df bcopy[nh][kk];
    v2df acopy[kk][kh];
    v2df ccopy[kk][kh];
    v2df acopy2[kk][kk];
    unsigned long bpcount, apcount, dotcount;
    bpcount= apcount= dotcount=0;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy[j][k] = *((v2df*)(b[k]+j+j));
    //    printf("copy b end\n");
    for(i=0;i<m;i+=kk){
	for(k=0;k<kk;k++){
	    __builtin_prefetch(a+i+k+8,0,0);
	    __builtin_prefetch(c+i+k+8,1,0);
	    v2df  *ak = (v2df*)(a[i+k]);
	    v2df * awp = (v2df*)(acopy+k);
	    v2df  *ck = (v2df*)(c[i+k]);
	    v2df * cwp = (v2df*)(ccopy+k);
	    awp[0]=ak[0];
	    awp[1]=ak[1];
	    awp[2]=ak[2];
	    awp[3]=ak[3];
	    cwp[0]=ck[0];
	    cwp[1]=ck[1];
	    cwp[2]=ck[2];
	    cwp[3]=ck[3];
	}
	for (j=0;j<n;j++){
	    double * ap = (double*)( acopy+j);
	    for (k=0;k<kk;k++){
		acopy2[j][k]=(v2df){ap[k],ap[k]};
	    }
	}
	//	printf("copy a end\n");
	for(k=0;k<kk;k++){
	    v2df * cp = (v2df*) ccopy[k];
	    v2df * ap = acopy2[k];
	    for(j=0;j<nh;j++){
		v2df * bp = bcopy[j];
		cp[j] -= ap[0]*bp[0]
		    + ap[1]*bp[1]
		    + ap[2]*bp[2]
		    + ap[3]*bp[3]
		    + ap[4]*bp[4]
		    + ap[5]*bp[5]
		    + ap[6]*bp[6]
		    + ap[7]*bp[7];
	    }
	}
	for(k=0;k<kk;k++){
	    v2df  *ck = (v2df*)(c[i+k]);
	    v2df * cwp = (v2df*)(ccopy+k);
#if 0	    
	    ck[0] = cwp[0];
	    ck[1] = cwp[1];
	    ck[2] = cwp[2];
	    ck[3] = cwp[3];
#endif	    
	    __builtin_ia32_movntpd((double*)(ck),cwp[0]);
	    __builtin_ia32_movntpd((double*)(ck+1),cwp[1]);
	    __builtin_ia32_movntpd((double*)(ck+2),cwp[2]);
	    __builtin_ia32_movntpd((double*)(ck+3),cwp[3]);
	}
	//	printf("calc c end\n");
    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
}
void matmul_for_nk8(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j,k;
    const int kk = 8;
    const int kh = kk/2;
    int nh = n/2;
    v2df bcopy[nh][kk];
    v2df acopy2[kk][kk];
    //    unsigned long bpcount, apcount, dotcount;
    //    bpcount= apcount= dotcount=0;
    BEGIN_TSC;
    
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy[j][k] = *((v2df*)(b[k]+j+j));
    //    END_TSC(bpcount);
    //    printf("copy b end\n");
#pragma omp parallel for private(i,j,k,acopy2)	 schedule(static)
    for(i=0;i<m;i+=kk){
	//	BEGIN_TSC;
	for(k=0;k<kk;k++){
	    __builtin_prefetch(a+i+k+16,0,0);
	    __builtin_prefetch(c+i+k+16,1,0);
	}
	for (j=0;j<n;j++){
	    double * ap = (double*)( a[i+j]);
	    for (k=0;k<kk;k++){
		acopy2[j][k]=(v2df){ap[k],ap[k]};
	    }
	}
	//	END_TSC(apcount);
	//	printf("copy a end\n");
	//	BEGIN_TSC;
	for(k=0;k<kk;k++){
	    v2df * cp = (v2df*) (c[i+k]);
	    v2df * ap = acopy2[k];
	    for(j=0;j<nh;j++){
		v2df * bp = bcopy[j];
		cp[j] -= ap[0]*bp[0] + ap[1]*bp[1]
		    + ap[2]*bp[2]    + ap[3]*bp[3]
		    + ap[4]*bp[4]    + ap[5]*bp[5]
		    + ap[6]*bp[6]    + ap[7]*bp[7];
	    }
	}
	//	printf("calc c end\n");
	//	END_TSC(dotcount);
    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //   (double)bpcount, (double)apcount, (double)dotcount);
    END_TSC(t,10);
}

void matmul_for_nk8_3(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{
    int ii;
    int dm = (m+31)/32;
    dm*= 8;
#pragma omp parallel for private(ii)	 schedule(static)
    for(ii=0;ii<4;ii++){
	int ifirst, iend;
	ifirst = ii*dm;
	iend = ifirst+dm;
	if (iend > m) iend = m;
	//	fprintf(stderr, "m, i, ifirst, iend = %d %d %d %d\n", m, ii, ifirst, iend);
	if (ifirst < m){
	    matmul_for_nk8_0(n1, a[ifirst], n2, b, n3, c[ifirst], iend-ifirst, n);
	}
    }
	
}

void matmul_for_nk16_0(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{
    int i;
    int mm=64;
    for(i=0;i<m;i+=mm){
	if (i+mm >m) mm=m-i;
	matmul_for_nk8_0(n1, (double(*)[]) (a[i]), n2, b,
			 n3, (double(*)[]) (c[i]), mm, 16);
	matmul_for_nk8_0(n1, (double(*)[]) (&a[i][8]), n2,(double(*)[])(b[8]),
			 n3, (double(*)[]) (c[i]), mm, 16);
    }
    
}
void matmul_for_nk16(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{
    if (m < 64){
	matmul_for_nk16_0c(n1, a, n2, b, n3, c, m, n);
	return;
    }
    int ii;
    int dm = (m+63)/64;
    dm*= 16;
#pragma omp parallel for private(ii)	 schedule(static)
    for(ii=0;ii<4;ii++){
	int ifirst, iend;
	ifirst = ii*dm;
	iend = ifirst+dm;
	if (iend > m) iend = m;
	//	fprintf(stderr, "m, i, ifirst, iend = %d %d %d %d\n", m, ii, ifirst, iend);
	if (ifirst < m){
	    matmul_for_nk16_0c(n1, a[ifirst], n2, b, n3, c[ifirst], iend-ifirst, n);
	}
    }
	
}

void matmul_for_nk32(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{
    int ii;
    int dm = (m+127)/128;
    dm*= 32;
#pragma omp parallel for private(ii)	 schedule(static)
    for(ii=0;ii<4;ii++){
	int ifirst, iend;
	ifirst = ii*dm;
	iend = ifirst+dm;
	if (iend > m) iend = m;
	//	fprintf(stderr, "m, i, ifirst, iend = %d %d %d %d\n", m, ii, ifirst, iend);
	if (ifirst < m){
	    matmul_for_nk32_0(n1, a[ifirst], n2, b, n3, c[ifirst], iend-ifirst, n);
	}
    }
	
}


void matmul_for_small_nk_7(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,j;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    unsigned long bpcount, apcount, dotcount;
    if (kk == 8){
	matmul_for_nk8(n1,  a, n2, b, n3, c, m, n);
	return;
    }
    BEGIN_TSC;
    bpcount= apcount= dotcount=0;
    //    BEGIN_TSC;
    for(k=0;k<kk;k++)
	for(j=0;j<nh;j++)
		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    //    END_TSC(bpcount);
    for(i=0;i<m;i+=2){
	//	BEGIN_TSC;
	double *ap=a[i];
	double *ap2=a[i+1];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df * cp2 = (v2df*) (&(c[i+1][0]));

	
	for(k=0;k<kk;k+=2){
	    v2df * aa = (v2df*)(ap+k);
	     __builtin_prefetch((double*)a[i+4]+k,0);
	     __builtin_prefetch((double*)a[i+5]+k,0);
	    acopy [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    aa = (v2df*)(ap2+k);
	    acopy2 [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy2[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    //   acp[k*2]=acp[k*2+1]=ap[k];
	    //	    acp[k*2+2]=acp[k*2+3]=ap[k+1];
	    //	    acp2[k*2]=acp2[k*2+1]=ap2[k];
	    //	    acp2[k*2+2]=acp2[k*2+3]=ap2[k+1];
	}
	//	END_TSC(apcount);
	//	BEGIN_TSC;
	for(j=0;j<nh;j++){
	    tmp = tmp2=  (v2df){0.0,0.0};
	    v2df ctmp= cp[j];
	    v2df ctmp2 = cp2[j] ;

	    v2df * bp = bcopy2[j];
	    __builtin_prefetch(c[i+4]+j,0);
	    __builtin_prefetch(c[i+5]+j,0);
	    for(k=0;k<kk;k+=8){
		int k2 = k+4;
		v2df *avp = acopy+k;
		v2df *avp2 = acopy2+k;
		v2df *bvp = bp+k;
		tmp += avp[0]*bvp[0];
		tmp2 += avp2[0]*bvp[0];
		tmp +=avp[1]*bvp[1];
		tmp2+=avp2[1]*bvp[1];
		tmp +=avp[2]*bvp[2];
		tmp2+=avp2[2]*bvp[2];
		tmp +=avp[3]*bvp[3];
		tmp2+=avp2[3]*bvp[3];
		tmp += avp[4]*bvp[4];
		tmp2 += avp2[4]*bvp[4];
		tmp +=avp[5]*bvp[5];
		tmp2+=avp2[5]*bvp[5];
		tmp +=avp[6]*bvp[6];
		tmp2+=avp2[6]*bvp[6];
		tmp +=avp[7]*bvp[7];
		tmp2+=avp2[7]*bvp[7];
	    }
	    
	    cp[j] = ctmp -tmp;
	    cp2[j] = ctmp2 -tmp2;

	}
	//	END_TSC(dotcount);

    }
    //    printf("m, kk, n = %d %d %d counts = %g %g  %g\n", m,kk,n,
    //	   (double)bpcount, (double)apcount, (double)dotcount);
    END_TSC(t,11);
    
}


void matmul_for_small_nk(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int kk,
			 int n)
{

    int i,ii;
    int nh = n/2;
    register int k;
    double bcopy[n][kk];
    v2df bcopy2[nh][kk];
    v2df acopy[kk];
    v2df acopy2[kk];
    double *acp = (double*) acopy;
    double *acp2 = (double*) acopy2;
    if (kk == 8){
	matmul_for_nk8_3(n1,  a, n2, b, n3, c, m, n);
	return;
    }
    if (kk == 16){
	matmul_for_nk16(n1,  a, n2, b, n3, c, m, n);
	return;
    }
    if (kk == 32){
	matmul_for_nk32(n1,  a, n2, b, n3, c, m, n);
	return;
    }
    BEGIN_TSC;
    for(k=0;k<kk;k++){
	int j;
	for(j=0;j<nh;j++)
	    bcopy2[j][k] = *((v2df*)(b[k]+j+j));
    }
#pragma omp parallel for private(i,k,acopy,acopy2)  schedule(static)
    for(i=0;i<m;i+=2){
	int j;
	double *ap=a[i];
	double *ap2=a[i+1];
	register v2df tmp, tmp2;
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df * cp2 = (v2df*) (&(c[i+1][0]));
	for(k=0;k<kk;k+=2){
	    v2df * aa = (v2df*)(ap+k);
	    acopy [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	    aa = (v2df*)(ap2+k);
	    acopy2 [k]= (v2df)  __builtin_ia32_shufpd(*aa,*aa,0x0);
	    acopy2[k+1]=(v2df) __builtin_ia32_shufpd(*aa,*aa,0xff);
	}
	__builtin_prefetch(a[i+4],0,3);
	__builtin_prefetch(c[i+4],1);
	__builtin_prefetch(a[i+5],0,3);
	__builtin_prefetch(c[i+5],1);
	__builtin_prefetch(a[i+20],0,3);
	__builtin_prefetch(c[i+20],1);
	__builtin_prefetch(a[i+21],0,3);
	__builtin_prefetch(c[i+21],1);
	for(j=0;j<nh;j++){
	    tmp = tmp2=  (v2df){0.0,0.0};
	    v2df ctmp= cp[j];
	    v2df ctmp2 = cp2[j] ;
	    v2df * bp = bcopy2[j];
	    for(k=0;k<kk;k+=8){
		int k2 = k+4;
		v2df *avp = acopy+k;
		v2df *avp2 = acopy2+k;
		v2df *bvp = bp+k;
		tmp += avp[0]*bvp[0];
		tmp2 += avp2[0]*bvp[0];
		tmp +=avp[1]*bvp[1];
		tmp2+=avp2[1]*bvp[1];
		tmp +=avp[2]*bvp[2];
		tmp2+=avp2[2]*bvp[2];
		tmp +=avp[3]*bvp[3];
		tmp2+=avp2[3]*bvp[3];
		tmp += avp[4]*bvp[4];
		tmp2 += avp2[4]*bvp[4];
		tmp +=avp[5]*bvp[5];
		tmp2+=avp2[5]*bvp[5];
		tmp +=avp[6]*bvp[6];
		tmp2+=avp2[6]*bvp[6];
		tmp +=avp[7]*bvp[7];
		tmp2+=avp2[7]*bvp[7];
	    }
	    cp[j] = ctmp -tmp;
	    cp2[j] = ctmp2 -tmp2;
	}
    }
    END_TSC(t,11);
}



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
	     int nc)
{
    double t0, t1, t2;
    if (k>= 512){
	get_cputime(&t0,&t1);
    }
    BEGIN_TSC;
    BEGIN_TIMER(timer);
#ifdef USEGDR
    if ((k>512)  || ((k==512) && ((n>=1024)||(m>=1024)))){
	//    if (k>=2048){
	mygdrdgemm(m, n, k, alpha, a, na, b, nb, beta, c, nc);
    }else{
	if ((k<=16) && (alpha == -1.0) && (beta == 1.0)){
	    matmul_for_small_nk(na, a,  nb, b, nc, c,  m, n, k);
	}else{
	    
	    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
			 m,n, k, alpha, a, na, b, nb, beta, c, nc);
	}
    }
#else
    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
	         m,n, k, alpha, a, na, b, nb, beta, c, nc);
#endif	
    //    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
    //		 m,n, k, alpha, a, na, b, nb, beta, c, nc);
    if (k==2048){
	END_TIMER(timer,31,((double)(m))*n*k*2);
	END_TSC(t,14);
    }else if (k==1024){
	END_TIMER(timer,32,((double)(m))*n*k*2);
	END_TSC(t,15);
    }else if (k==512){
	END_TIMER(timer,33,((double)(m))*n*k*2);
	END_TSC(t,17);
    }else{
	END_TIMER(timer,34,((double)(m))*n*k*2);
	END_TSC(t,18);
    }
	
    if (k>= 512){
	get_cputime(&t0,&t1);
	dprintf(10,"dgemm M=%d N=%d K=%d time=%10.4g  %g Gflops\n",
		m,n,k,t0, ((double)m)*n*k*2/t0/1e9);
    }
}


void reset_gdr(int m, double a[][m], int nb, double awork[][nb], int n)
{
#ifdef USEGDR    
    double aw2[nb][nb];
    if (nb < 2048){
	fprintf(stderr,"reset_gdr nb = %d <2048 not supported\n", nb);
	exit(-1);
    }
    gdr_check_and_restart(a, awork, aw2);
    
    int i,j;
    dprintf(9,"reset_gdr clear awork\n");
    for (i=0;i<nb;i++){
	for (j=0;j<n;j++){
	    awork[j][i]=0;
	}
    }
    dprintf(9,"reset_gdr clear aw2\n");
    for (i=0;i<nb;i++){
	for (j=0;j<nb;j++){
	    aw2[j][i]=0;
	}
    }
    dprintf(9,"reset_gdr try_dgemm\n");
    gdrsetforceswapab();    
    cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans,
	         n,nb, nb, 1.0, awork, nb, aw2, nb, 0.0, a, m);
    mydgemm(n,nb,nb,1.0,awork,nb,aw2,nb,0.0,a,m);

#endif    
}

#ifndef USEGDR
void gdrsetforceswapab(){}
void gdrresetforceswapab(){}
void gdrsetskipsendjmat(){};
void gdrresetskipsendjmat(){}
void gdrsetnboards(){}
void set_matmul_msg_level(int level){}
void gdrdgemm_set_stress_factor(int x){}
#endif
