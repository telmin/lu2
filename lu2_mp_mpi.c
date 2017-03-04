//
// lu2_mp_mpi.c
//
#define HUGEPAGE
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdarg.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>

#include <emmintrin.h>
#include "mpi.h"
#include "lu2_mp.h"
MPI_Status status;

static int local_proc_id;
static int total_proc_count;
static char local_proc_name[MPI_MAX_PROCESSOR_NAME];
typedef struct mpidoubleint{
    double val;
    int index;
}MPIDOUBLEINT,  *PMPIDOUBLEINT;
    

int MP_myprocid()
{
    return local_proc_id;
}

int MP_proccount()
{
    return total_proc_count;
}


void MP_bcast(void * p, int size)
{
    MPI_Bcast(p,size,MPI_BYTE,0,MPI_COMM_WORLD);
}


void MP_initialize(int * argc,  char *** argv)
{
    int  namelen;
    int myid, numprocs;
    int retval;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    //    MPI_Init(argc, argv);
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &retval);
    //    MPI_Init_thread(argc, argv, MPI_THREAD_SINGLE, &retval);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);
    //    strcpy(local_proc_name, processor_name);
    fprintf(stderr, "Initialize:Myid = %d Myname = %s Nprocs=%d threads=%d\n",
	    myid, processor_name, numprocs,retval);
    fprintf(stderr,"MP_initialize: omp_max_threads=%d procs=%d\n",
		omp_get_max_threads(),omp_get_num_procs());
    total_proc_count = numprocs;
    local_proc_id = myid;
    MPI_Barrier(MPI_COMM_WORLD);
}


void  MP_sync()
{
    //    if (local_proc_id == 0) cerr << "Enter MP_SYNC\n";
    MPI_Barrier(MPI_COMM_WORLD);
}


void MP_end()
{
        MPI_Finalize();
}

void MP_error(char * message)
{
    fprintf(stderr,"Proc id = %d message = %s\n", local_proc_id, message);
    exit (-1);
}
void MP_message(char * message)
{
    fprintf(stderr,"Proc id = %d message = %s\n", local_proc_id, message);
}

static int hfd;
void MP_allocate_hugetlbfs(char * name,
			  void ** a,
			  void ** ac,
			  void ** ac2,
			  void ** dinv,
			  void ** ar,
			  void ** ar2,
			  int n,
			  int nb,
			  int n2)
{
    // seemingly does not work with current mgv (kernel 2.6.35.13, OFED 1.5.3.1?)
    //
    char work[255];
    sprintf(work,"%sMP-%d",name,local_proc_id);
    int fd= open(work, O_RDWR|O_CREAT, 0777);
    hfd = fd;
    long nl = n;
    long nbl = nb;
    long nbbl = nb+32;
    long n2l = n2;
    int nbb = nb+32;
    if (fd == -1) MP_error("open hugetlbfs");
    size_t size = ((long)(sizeof(double)*((long)n)*(long)(n2))+0x400000)
	&0xffffffffffc00000L;
    *a = mmap(0, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
    size_t worksize = (((long)sizeof(double)*nbbl*nl)+0x400000)&0xffffffffffc00000L;
    off_t offset = (off_t) size;
    *ac = mmap(0, worksize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
    offset += (off_t) worksize;
    *ac2 = mmap(0, worksize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
    offset += (off_t) worksize;
    *dinv = mmap(0, worksize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
    offset += (off_t) worksize;
    worksize = (((long)sizeof(double))*nbl*nbbl*5+0x400000L)&0xffffffffffc00000L;
    //    worksize = ((sizeof(double)*nb*n2)+0x400000)&0xffc00000;
    *ar = mmap(0, worksize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);
    offset += (off_t) worksize;
    *ar2 = mmap(0, worksize, PROT_READ|PROT_WRITE, MAP_SHARED, fd, offset);

    //    *a = (double(*)[]) malloc(sizeof(double)*nl*n2l);
    //*ac = (double(*)[]) malloc(sizeof(double)*n*nbb);
    //    *ac2 = (double(*)[]) malloc(sizeof(double)*n*nbb);
    //    *dinv = (double(*)[]) malloc(sizeof(double)*n*nbb);
    //    *ar = (double*) malloc(sizeof(double)*nb*nbb*8);
    //    *ar2 = (double*) malloc(sizeof(double)*nb*nbb*8);

    
    fprintf(stderr, "procid=%d *a, *ar2  size offset worksize= %lx %lx %lx %lx %lx %lx %lx %lx\n",
	    local_proc_id,
    	   (long) (*a), (long) (*ac),(long) (*ac2),(long) (*ar),(long) (*ar2),
	    (long) (size), (long) offset, (long) worksize);
}

void  MP_setup_communicators(PPARMS parms, PCONTROLS controls)
{
    int rank_in_row, rank_in_col;
    if (parms->procs_row_major){
	rank_in_row = local_proc_id % parms->npcol;
	rank_in_col = local_proc_id / parms->npcol;
    }else{
	rank_in_row = local_proc_id / parms->nprow;
	rank_in_col = local_proc_id % parms->nprow;
    }
    dprintf(9,"setup_communicator, myid, rowid, colid = %d %d %d\n",
	    local_proc_id, rank_in_row, rank_in_col);
    
    if (MPI_Comm_split( MPI_COMM_WORLD, rank_in_row,
			rank_in_col, &(controls->col_comm)) != MPI_SUCCESS)
	MP_error("Comm_split failed");
    if (MPI_Comm_split( MPI_COMM_WORLD, rank_in_col,
			rank_in_row, &(controls->row_comm)) != MPI_SUCCESS)
	MP_error("Comm_split failed");
    controls->pid = local_proc_id;
    controls-> rank_in_row = rank_in_row;
    controls-> rank_in_col = rank_in_col;
    int rrow, rcol;
    MPI_Comm_rank(controls-> row_comm,&rrow);
    MPI_Comm_rank(controls-> col_comm,&rcol);
    dprintf(9,"setup_communicator after split: myid, rowid, colid = %d %d %d\n",
	    local_proc_id, rrow, rcol);

}

void MP_max_and_maxloc(double * val, int * pid, MPI_Comm mpi_comm)
{
    MPIDOUBLEINT in;
    MPIDOUBLEINT out;
    in.val=*val;
    in.index=*pid;
    //    dprintf(9,"call allreduce\n");
    MPI_Allreduce(&in,&out, 1, MPI_DOUBLE_INT, MPI_MAXLOC,mpi_comm);
    //    dprintf(9,"call allreduce end\n");
    *val = out.val;
    *pid = out.index;
}

#define RING_BCAST_SIZE 1024000
//#define USEMPIBCAST    
static void divided_singlebcast(double * p, int size, int source, MPI_Comm comm,
				int myid, int numprocs, int message_id)
{
#ifdef USEMPIBCAST
    //    dprintf(9,"divided_singlebcast, size=%d\n", size);
    //    MPI_Barrier(comm);
    //    dprintf(9,"divided_singlebcast, barrier end\n");
    MPI_Bcast(p,size*sizeof(double),MPI_BYTE,source,comm);
    //    dprintf(9,"divided_singlebcast, end, size=%d\n", size);
#else
    MPI_Status status;
    int mypos = (myid+numprocs-source)%numprocs;
    int left = (myid - 1 + numprocs)%numprocs;
    int right = (myid  + 1)%numprocs;
    //    dprintf(9,"source=%d, mypos=%d left=%d right=%d\n",source, mypos, left,right);
    if (mypos > 0){
	//	dprintf(9,"recieve data %d %d %d\n", size, left,message_id);
	MPI_Recv(p,size,MPI_DOUBLE,left,message_id,comm, &status);
	//	dprintf(9,"recieve data end %d %d %d\n", size, left,message_id);
    }
    if (mypos < numprocs -1){
	//	dprintf(9,"send data %d %d %d\n", size, right,message_id);
	MPI_Send(p,size,MPI_DOUBLE,right,message_id,comm);
	//	dprintf(9,"send data end %d %d %d\n", size, right,message_id);
    }
#endif    
}
	
    
static void divided_bcast(double * p, int size, int source, MPI_Comm comm)
{
    int i, myid,numprocs;
    MPI_Comm_rank(comm,&myid);
    MPI_Comm_size(comm,&numprocs);
    dprintf(9,"numprocs = %d\n", numprocs);
    int message_id=MPMYBCASTTAG;
    for(i=0;i<size;i+= RING_BCAST_SIZE){
	int buffsize= RING_BCAST_SIZE;
	if (i+buffsize > size) buffsize = size-i;
	divided_singlebcast( p+i, buffsize, source, comm,myid,
			     numprocs,message_id);
	message_id++;
    }
}

void MP_mybcast(void * p, int size, int source, MPI_Comm comm)
{
    //    MPI_Barrier(comm);
    //    dprintf(9,"size=%d\n", size);
    //    print_current_time("enter ring_bcast");
    divided_bcast((double*)p,size/sizeof(double),source,comm);
    //    print_current_time("end   ring_bcast");
    //    MPI_Barrier(comm);
}

static FILE* matoutfid;

void init_matoutfid()
{
    char work[255];
    sprintf(work,"%sMP-%d","/tmp/matdata",local_proc_id);
    matoutfid = fopen(work,"w");
}


void printf_matfile(char *  fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);	
    vfprintf(matoutfid, fmt, ap);
    va_end(ap);
}


void print_mat_to_matoutfid(char * name,
			    int nncol,
			    int m,
			    int n,
			    double mat[][nncol])
{
    int i, j;
    for(i=0;i<m;i++){
	for(j=0;j<n;j++){
	    fprintf(matoutfid," %s(%d,%d)=%10.6e", name,i,j,mat[i][j]);
	    if ((j % 4) == 3)fprintf(matoutfid,"\n");
	}
	fprintf(matoutfid,"\n");
    }
}

			   
