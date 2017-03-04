//
// timerlib.c
//
// J. Makino
//    Time-stamp: <10/11/09 22:05:31 makino>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdio.h>
#include <time.h>

#ifdef MPI
#include "lu2_mp.h"
#endif

#define FTYPE double
#include <emmintrin.h>
typedef double v2df __attribute__((vector_size(16)));
typedef union {v2df v; double s[2];}v2u;

#define TIMER_MAX (100)
static double timers[TIMER_MAX];
static double ops[TIMER_MAX];

void init_timer()
{
    int i;
    for(i=0;i<TIMER_MAX;i++)timers[i]=0;
}

void accum_timer(double val, int index)
{
    timers[index] += val;
}
void accum_timer_and_ops(double val, int index, double opsval)
{
    timers[index] += val;
    ops[index] += opsval;
}

void print_a_timer(char * name, int index, double ops)
{
    printf("%s time=%16.6g ops/cycle=%g\n", name, timers[index], ops/timers[index]);
}
void print_timers(double n, double nb)
{
    print_a_timer("swaprows   ", 0, n*n/2);
    print_a_timer("scalerow   ", 1, n*n/2);
    print_a_timer("trans rtoc ", 2,n*n/2);
    print_a_timer("trans ctor ", 3,n*n/2);
    print_a_timer("trans mmul ", 4,n*n/4*(30));
    print_a_timer("tr nr  cdec", 5,n*n/2);
    print_a_timer("trans vvmul", 6,n*n/2);
    print_a_timer("trans findp", 7,n*n/2);
    print_a_timer("solve tri u", 8,n);
    print_a_timer("solve tri  ", 9,n*n*nb);
    print_a_timer("matmul nk8 ", 10,n*n/4*16);
    print_a_timer("matmul snk ", 11,n*n/4*32);
    print_a_timer("trans mmul8", 12,n*n/4*16);
    print_a_timer("trans mmul4", 13,n*n/4*8);
    print_a_timer("trans mmul2", 16,n*n/4*4);
    print_a_timer("DGEMM2K    ", 14,n*n*(n/3.0*2.0-nb/2.0));
    print_a_timer("DGEMM1K    ", 15,n*n*512.0);
    print_a_timer("DGEMM512   ", 17,n*n*256.0);
    print_a_timer("DGEMMrest  ", 18,n*n*256.0);
    print_a_timer("col dec t  ", 20,n*n*16.0);
    print_a_timer("Total      ", 19,n*n*(n*2.0/3.0));
}

#ifdef MPI
void print_a_timer_mp(char * name, int index)
{
    dprintf(1,"%-25s time=%16.6g ops/cycle=%12.6g\n", name, timers[index],
	    ops[index]/timers[index]);
}

void print_timers_mpi(int pid)
// pid >= 0 : only pid prints
//     <0   : all print
{
    int proccount =MP_proccount();
    int ii, i, j;
    MPI_Barrier(MPI_COMM_WORLD);
    for (ii=0;ii<proccount; ii++){
	if ((MP_myprocid() == ii) && ((pid < 0) || (pid == ii))){
	    //	    print_a_timer_mp("Left bcast etc", 0);
	    print_a_timer_mp("update", 1);
	    print_a_timer_mp("update matmul", 2);
	    print_a_timer_mp("update swap+bcast", 3);
	    print_a_timer_mp("total", 4);
	    print_a_timer_mp("rfact", 5);
	    print_a_timer_mp("ldmul     ", 6);
	    print_a_timer_mp("colum dec with trans", 7);
	    print_a_timer_mp("colum dec right ", 8);
	    print_a_timer_mp("colum dec left ", 9);
	    print_a_timer_mp("rowtocol ", 10);
	    print_a_timer_mp("column dec in trans", 11);
	    print_a_timer_mp("coltorow  ", 12);
	    print_a_timer_mp("dgemm8    ", 13);
	    print_a_timer_mp("dgemm16   ", 20);
	    print_a_timer_mp("dgemm32   ", 21);
	    print_a_timer_mp("dgemm64   ", 22);
	    print_a_timer_mp("dgemm128  ", 23);
	    print_a_timer_mp("main dgemm", 19);
	    print_a_timer_mp("col trsm  ", 24);
	    print_a_timer_mp("col update  ", 25);
	    print_a_timer_mp("col r dgemm", 26);
	    print_a_timer_mp("col right misc ", 27);
	    print_a_timer_mp("backsub ", 28);
	    print_a_timer_mp("col dec total ", 30);
	    print_a_timer_mp("DGEMM2k ", 31);
	    print_a_timer_mp("DGEMM1k ", 32);
	    print_a_timer_mp("DGEMM512 ", 33);
	    print_a_timer_mp("DGEMMrest ", 34);
	    print_a_timer_mp("TRSM U ", 35);
 	    print_a_timer_mp("col r swap/scale", 36);
 	    print_a_timer_mp("rfact ex coldec", 37);
 	    print_a_timer_mp("ld_phase1", 38);
 	    print_a_timer_mp("rfact misc1", 39);
 	    print_a_timer_mp("rfact misc2", 40);
 	    print_a_timer_mp("rfact misc3", 41);
 	    print_a_timer_mp("rfact misc4", 42);
 	    print_a_timer_mp("copysubmats", 43);
	    fflush(stdout);
	}
	MPI_Barrier(MPI_COMM_WORLD);
    }
}
    
#endif

static double walltime0;


void get_cputime(double * laptime, double *splittime)
{
	struct timeval tval;
	struct timezone tz;

	gettimeofday(&tval,&tz);
	*laptime = tval.tv_sec + tval.tv_usec * 1.0e-6 - *splittime;
	*splittime = tval.tv_sec + tval.tv_usec * 1.0e-6 ;
}

void init_current_time()
{
    double time1, time0;
    get_cputime(&time1, &walltime0);
}

static int max_pid_for_print_time=0;
void set_max_pid_for_print_time(int p)
{
    max_pid_for_print_time=p;
}
void print_current_time(char * message)
{
    double time1, time0;
    if (max_pid_for_print_time== 0 || MP_myprocid()< max_pid_for_print_time){
	get_cputime(&time1, &time0);
	fprintf(stderr,"P%d, %s, time=%g\n",
		MP_myprocid(), message, time0-walltime0);
    }
}

void print_current_datetime(char * message)
{
    struct timeval tval;
    struct timezone tz;
    
    gettimeofday(&tval,&tz);
    char * ctimep;
    ctimep = ctime(&tval.tv_sec);
    printf("\n%s %s\n", message, ctimep);
}
