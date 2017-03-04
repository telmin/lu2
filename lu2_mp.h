//
// lu2_mp.h
//

#ifndef ___LU2_MP___
#define ___LU2_MP___

#include "mpi.h"

#define MPSWAPTAG 1
#define MPBSTAG 2
#define MPRCOMMTAG 1024
#define MPMYBCASTTAG 2048
#define MPSENDBASETAG 2048
#define MPNRMESSAGETAG 4096
#define NNCOLINC 32
#define NRMESSAGES 1024

#define MAXRCOMMMESSAGE 1048576
typedef struct parmstruct{
    int n;
    int seed;
    int nb;
    int nprow;
    int npcol;
    int procs_row_major;
    int usehugepage;
    int twocards;
    int ncards;
    int firstcard;
    int maxpidfortiming;
    int vcommscheme;
    int stress_factor;
} PARMS, *PPARMS;

typedef struct control_struct{
    MPI_Comm row_comm;
    MPI_Comm  col_comm;
    int nrow;
    int ncol;
    int nnrow;
    int nncol;
    int pid;
    int rank_in_row;
    int rank_in_col;
    int check_rcomm_transfer;
} CONTROLS, *PCONTROLS;


enum MESSAGE_STATE{UNUSED,INITIALIZED,RECEIVING, RECEIVED, SENDING, SENT};
typedef struct message_struct{
    void * mptr;
    int length;
    enum MESSAGE_STATE message_state;
    MPI_Request request;
}MESSAGE, *PMESSAGE;

typedef struct rcomm_struct{
    int nextp;
    int prevp;
    int first;
    int last;
    int nmessages;
    int preceive;
    int psend;
    MPI_Comm comm;
    MESSAGE message[NRMESSAGES];
}RCOMMT, *PRCOMMT;
    
int MP_myprocid();
int MP_proccount();
void MP_bcast(void * p, int size);
void MP_initialize(int * argc, char *** argv);
void  MP_sync();
void MP_end();
void MP_error(char * message);
void MP_message(char * message);
void MP_allocate_hugetlbfs(char * name,
			  void ** a,
			  void ** ac,
			  void ** ac2,
			  void ** dinv,
			  void ** ar,
			  void ** ar2,
			  int n,
			  int nb,
			  int n2);
void  MP_setup_communicators(PPARMS parms, PCONTROLS controls);
void MP_max_and_maxloc(double * val, int * pid, MPI_Comm mpi_comm);
void MP_mybcast(void * p, int size, int source, MPI_Comm comm);

int pcolid(int i, PPARMS parms, PCONTROLS  controls);
int have_current_col(int i, PPARMS parms, PCONTROLS  controls);
int first_row(int i, PPARMS parms, PCONTROLS  controls);
int local_colid(int i, PPARMS parms, PCONTROLS  controls);
int global_rowid(int i, PPARMS parms, PCONTROLS  controls);
int global_colid(int i, PPARMS parms, PCONTROLS  controls);

void convert_global_col_range_to_local_range(int c1, int c2,
					     int* lc1, int *lc2,
					     PPARMS parms, PCONTROLS controls);

void init_matoutfid();

void print_mat_to_matoutfid(char * name,
			    int nncol,
			    int m,
			    int n,
			    double mat[][nncol]);
void printf_matfile(char *  fmt, ...);


void swapvoidptr(void **a, void **b);
void swapdoubleptr(double **a, double **b);

#endif // ___LU2_MP___

