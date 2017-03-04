//
// gemmtest.c
//
// J. Makino
//    Time-stamp: <10/10/17 13:43:33 makino>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <timerlib.h>
#ifndef NOBLAS
#include <cblas.h>
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

void daxpy(v2df a[], v2df b[], v2df c[], int n)
{
    int i;
    for(i=0;i<n; i+=8){
	b[i] += a[i]*c[0];
	b[i+1] += a[i+1]*c[0];
	b[i+2] += a[i+2]*c[0];
	b[i+3] += a[i+3]*c[0];
	b[i+4] += a[i+4]*c[0];
	b[i+5] += a[i+5]*c[0];
	b[i+6] += a[i+6]*c[0];
	b[i+7] += a[i+7]*c[0];
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
    for(j=0;j<nh;j++)
	for(k=0;k<kk;k++)
	    //		bcopy2[j][k] = *((v2df*)(b[k]+j+j));
	    bcopy2[j][k] = *(((v2df*)b)+j*n2+k);
    for(i=0;i<m;i+=2){
	//	BEGIN_TSC;
	v2df *ap = (v2df*) a[i];
	v2df * cp = (v2df*) (&(c[i][0]));
	v2df *app = (v2df*) a[i+1];
	v2df * cpp = (v2df*) (&(c[i+1][0]));
	
	int k;
	
	for(j=0;j<nh;j+=2){
	    v2df * bvp0 = bcopy2[j];
	    v2df * bvp1 = bcopy2[j+1];
	    LOADPD(cp[j],X12);
	    LOADPD(cp[j+1],X13);
	    LOADPD(cpp[j],X14);
	    LOADPD(cpp[j+1],X15);

	    for(k=0;k<8;k++){
		
		LOADPD(ap[k],X0);
		LOADPD(app[k],X2);
		LOADPD(bvp0[k*2],X4);
		LOADPD(bvp1[k*2],X5);
		LOADPD(bvp0[k*2+1],X8);
		LOADPD(bvp1[k*2+1],X9);
		MOVAPD(X4,X6);
		MOVAPD(X5,X7);
		MOVAPD(X8,X10);
		MOVAPD(X9,X11);
		MOVAPD(X0,X1);
		BCAST0(X0);
		BCAST1(X1);
		MOVAPD(X2,X3);
		BCAST0(X2);
		BCAST1(X3);
		MULPD(X0,X4);
		MULPD(X0,X5);
		MULPD(X1,X8);
		MULPD(X1,X9);
		SUBPD(X4,X12);
		SUBPD(X5,X13);
		SUBPD(X8,X12);
		SUBPD(X9,X13);
		MULPD(X2,X6);
		MULPD(X2,X7);
		MULPD(X3,X10);
		MULPD(X3,X11);
		SUBPD(X6,X14);
		SUBPD(X7,X15);
		SUBPD(X10,X14);
		SUBPD(X11,X15);
	    }

	    STORPD(X12,cp[j+0]);
	    STORPD(X13,cp[j+1]);
	    STORPD(X14,cpp[j+0]);
	    STORPD(X15,cpp[j+1]);
	}
    }
}
void matmul_for_nk16_test1(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{

    int i,j,k;
    int kk = 16;
    int nh = n/2;
    v2df bcopy[nh][kk];
    for(j=0;j<nh;j++)
	for(k=0;k<kk;k++)
		bcopy[j][k] = *((v2df*)(b[k]+j+j));
    for(i=0;i<m;i+=2){
	
	v2df csub[nh][2];
	for(j=0;j<2;j++){
	    v2df * src = (v2df*) c[i+j];
	    for(k=0;k<nh;k++){
		csub[k][j]=src[k];
	    }
	}

	for(k=0;k<kk;k+=4){
	    v2df *ap = (v2df*) a[i]+k;
	    v2df *app = (v2df*) a[i+1]+k;
	    LOADPD(ap[0],X8);
	    LOADPD(ap[1],X9);
	    LOADPD(app[0],X10);
	    LOADPD(app[1],X11);
	    MOVAPD(X8,X0);	
	    BCAST0(X0);
	    MOVAPD(X8,X1);	
	    BCAST1(X1);
	    MOVAPD(X9,X2);	
	    BCAST0(X2);
	    MOVAPD(X9,X3);	
	    BCAST1(X3);
	    MOVAPD(X10,X4);	
	    BCAST0(X4);
	    MOVAPD(X10,X5);	
	    BCAST1(X5);
	    MOVAPD(X11,X6);	
	    BCAST0(X6);
	    MOVAPD(X11,X7);	
	    BCAST1(X7);
	    // 2x4 a, size doubled to 4x4
	    for(j=0;j<nh;j++){
		v2df * bvp = bcopy[j]+k;
		v2df * cvp = csub[k];
		LOADPD(bvp[0],X8);
		LOADPD(bvp[1],X9);
		
		MOVAPD(X8,X14);
		MOVAPD(X9,X15);
		MULPD(X0,X14);
		MULPD(X1,X15);
		
		LOADPD(bvp[2],X10);
		LOADPD(bvp[3],X11);

		
		LOADPD(cvp[0],X12);
		LOADPD(cvp[1],X13);

		SUBPD(X14,X12);
		SUBPD(X15,X12);
		MOVAPD(X10,X14);
		MOVAPD(X11,X15);
		MULPD(X2,X14);
		MULPD(X3,X15);
		SUBPD(X14,X12);
		SUBPD(X15,X12);
		MULPD(X4,X8);
		MULPD(X5,X9);
		MULPD(X6,X10);
		MULPD(X7,X11);
		SUBPD(X8,X13);
		SUBPD(X9,X13);
		SUBPD(X10,X13);
		SUBPD(X11,X13);
		STORPD(X12, cvp[0]);
		STORPD(X13, cvp[1]);
	    }
	}
	// copyback cmat
	for(j=0;j<2;j++){
	    v2df * dest = (v2df*) c[i+j];
	    for(k=0;k<nh;k++){
		dest[k] = csub[k][j];
	    }
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
}

void matmul_for_nk16(int n1, double a[][n1],
			 int n2, double b[][n2],
			 int n3, double c[][n3],
			 int m,
			 int n)
{
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
	    matmul_for_nk16_0a(n1, a[ifirst], n2, b, n3, c[ifirst], iend-ifirst, n);
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



#ifndef USEGDR
void gdrsetforceswapab(){}
void gdrresetforceswapab(){}
void gdrsetskipsendjmat(){};
void gdrresetskipsendjmat(){}
void gdrsetnboards(){}
void set_matmul_msg_level(int level){}
#endif



#define N 1024
#define K 16
int main()
{
    double a[N][K];
    double c[N][K];
    double b[K][K];
    int i,j;
    for(i=0;i<N;i++)
	for(j=0;j<K;j++){
	    a[i][j] = i*j;
	   c[i][j] = 0;
	}
    for(i=0;i<K;i++)
	for(j=0;j<K;j++)
	    b[i][j] = i*j;
    
    int k;
    unsigned long int start, end;
    rdtscl(&start);
#define NT 5000
    for(i=0;i<NT;i++){
	matmul_for_nk16_test1(K, a, K, b, K, c, N, K);
    }
    rdtscl(&end);
    printf("cycles = %e; %e ops/clock\n",
	   (double)(end-start), (N*K*K*(NT+0.0))/((double)(end-start)));
}

