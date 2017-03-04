/*
 * MPITEST.C
 *
 * Copyright  J. Makino March 2 2001
 *
 * JM's very first MPI program....
 */
#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include "lu2_mp.h"
#ifndef MPI
#define MPI
#endif
#ifdef TCPLIB
#include "tcplib.h"
#endif
#define MAXBUFSIZE 5000000
MPI_Status status;
double sendbuffer[MAXBUFSIZE];
double receivebuffer[MAXBUFSIZE];

void sendbuf(int size, int dest)
{
    MPI_Send( &sendbuffer, size, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}
void receivebuf(int size, int source)
{
    MPI_Recv( &receivebuffer, size, MPI_DOUBLE, source, 0,
	      MPI_COMM_WORLD, &status);
}

void initbuf(int size)
{
    int i;
    for (i=0;i<size;i++){
	sendbuffer[i] = i;
	receivebuffer[i]=99999;
    }
}
double f(double);

double f(double a)
{
    return (4.0 / (1.0 + a*a));
}

int main(int argc,char *argv[])
{
    int done = 0, n, myid, numprocs, i;
    int ringmax, interval;
    double PI25DT = 3.141592653589793238462643;
    double mypi, pi, h, sum, x;
    double startwtime = 0.0, endwtime;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int size, count, ic;
    int left, right, myoffset,level;
    //    MPI_Init(&argc,&argv);
    int retval;
    //    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &retval);
    MP_initialize(&argc,&argv);init_current_time();
    fprintf(stderr,"thred val=%d\n", retval);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);

    ringmax = 4;
    if (ringmax > numprocs) ringmax = numprocs;

    fprintf(stdout,"Process %d of %d on %s\n",
	    myid, numprocs, processor_name);

    if (numprocs == 1){
	fprintf(stderr,"I have only one procesor... nothing to do\n");
	MPI_Finalize();
	return 0;
    }
    n = 0;

    interval = numprocs/ringmax;
    left = (myid-interval+numprocs)%numprocs;
    right = (myid+interval)        %numprocs;
    fprintf(stderr,"Myid, R, L= %d %d %d\n",myid,right,left);
#ifdef TCPLIB
    tcp_request_full_MPI_connection();
#endif
    
#define SIZEMAX     5000000
#define COUNTMAX    2000
#define TOTALMAX   6e6
    count = COUNTMAX;
    if ( myid == 0){
	startwtime = MPI_Wtime();
    }
    for(ic=0;ic<count;ic++){
	MPI_Barrier(MPI_COMM_WORLD);
    }
    if (myid == 0){
	double dtime;
	endwtime = MPI_Wtime();
	dtime = endwtime - startwtime;
	printf("Barrier count = %d wall clock time = %f  %f us/synch\n",
	       count, dtime, (dtime*1e6)/count);	       
    }
    if ( myid == 0){
	startwtime = MPI_Wtime();
    }
    for(ic=0;ic<count;ic++){
	double ttmp = myid;
	double global_tmin;
	MPI_Allreduce(&ttmp, &global_tmin,1, MPI_DOUBLE, MPI_MIN,MPI_COMM_WORLD);
    }
    if (myid == 0){
	double dtime;
	endwtime = MPI_Wtime();
	dtime = endwtime - startwtime;
	printf("Allreduce count = %d wall clock time = %f  %f us/call\n",
	       count, dtime, (dtime*1e6)/count);	       
    }

#ifdef TCPLIB
    for(level = 2; level <= numprocs; level*=2){
	if ( myid == 0){
	    startwtime = MPI_Wtime();
	}
	for(ic=0;ic<count;ic++){
	    tcp_barrier(level);
	}
	tcp_barrier(numprocs);
	
	if (myid == 0){
	    double dtime;
	    endwtime = MPI_Wtime();
	    dtime = endwtime - startwtime;
	printf("TCP Barrier level %d count = %d wall clock time = %f  %f us/synch\n",
	       level,count, dtime, (dtime*1e6)/count);	       
	}
    }
    
    
    
    if ( myid == 0){
	startwtime = MPI_Wtime();
    }
    for(ic=0;ic<count;ic++){
	double ttmp = myid;
	double global_tmin= tcp_allmax(ttmp);
	if (global_tmin  != (numprocs - 1)){
	    fprintf(stderr,"tcp allmax failed myid %d %d %d\n",
		    myid, global_tmin, numprocs);
	}
    }
    if (myid == 0){
	double dtime;
	endwtime = MPI_Wtime();
	dtime = endwtime - startwtime;
	printf("TCP Allreduce count = %d wall clock time = %f  %f us/call\n",
	       count, dtime, (dtime*1e6)/count);	       
    }
    
#endif    
    for(size=1;size <= SIZEMAX; size *= 4){
	int mode;
	count = COUNTMAX;

#ifdef TCPLIB
#define MODEMAX 2
#define MODEMIN 0
#else
#define MODEMAX 3
#define MODEMIN 0
#endif	
	for(mode=MODEMIN;mode <MODEMAX;mode++){
	    int length = size*sizeof(double);
	    initbuf(size);
	    MPI_Barrier(MPI_COMM_WORLD);
	    startwtime = MPI_Wtime();
	    if (((double)count)*size > TOTALMAX) count = TOTALMAX/size;
	    for(ic=0;ic<count;ic++){
		
		if (mode == 0){
		    MPI_Sendrecv(&sendbuffer,size,MPI_DOUBLE,left,0,
				 &receivebuffer,size,MPI_DOUBLE,right,0,
				 MPI_COMM_WORLD, &status);
		}else if (mode == 1){
		    MPI_Bcast(&sendbuffer,size*8,MPI_BYTE,0,MPI_COMM_WORLD);
		    
		}else if (mode == 2){
		    MP_mybcast(&sendbuffer,size*8,0,MPI_COMM_WORLD);
		    
#if 0
		    if ((myid/(numprocs/ringmax)%2) == 0){
		    /*		if ((ic % 1000) == 0)fprintf(stderr,"count = %d\n", ic);*/
			sendbuf(size, left);
			receivebuf(size,right);
		    }else{
			receivebuf(size,right);
			sendbuf(size, left);
		    }
		}else{
		    int sendparms[2];
		    int receiveparms[2];
		    sendparms[0] = length;
		    tcp_simd_transfer_data_by_MPIname(left,sendparms,1,sendbuffer,
						      right,receiveparms,receivebuffer);
#endif		    
		}
	    }
	    if (myid == 0){
		double dtime;
		endwtime = MPI_Wtime();
		dtime = endwtime - startwtime;
		printf("size, count = %d %d wall clock time = %f  %f MB/s\n",
		       size, count, dtime,
		       ((double)size)*count*sizeof(double)/dtime/1e6*2);	       
		fflush( stdout );
	    }else if (myid==-1){
#if 1		
		if (mode )
		    for(i=0;i<size;i+= 1+size/3){
			fprintf(stderr," data[%d] = %e\n", i, receivebuffer[i]);
		    }
#endif		
	    }
	}
    }
    MPI_Finalize();
    return 0;
}
