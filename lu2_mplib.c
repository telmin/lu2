//
// lu2_mplib.c
//

#include <stdlib.h>
#include <stdio.h>

#include "lu2_mp.h"

#include <stdarg.h>

/*
 * DPRINTF: printf-style debugging messages, controlled by debug_level
 *	    as set by the user interface (debug=)
 */

static int debug_level = 99;

void set_debug_level(i)
{
    debug_level = i;
}

void mydprintf(int debug, char *  fmt, ...)
{
    va_list ap;


    if ((debug <= debug_level) || (debug==999)) {
	char s[8192];
	/* print this debug message? */
        va_start(ap, fmt);	
	vsprintf(s, fmt, ap);
        va_end(ap);

	fprintf(stderr,"Proc:%d lbl:%d %s",MP_myprocid(), debug,s);
	fflush(stderr);
    }
    if (debug==999){
	MPI_Barrier(MPI_COMM_WORLD);
	sleep(5);
	exit(1);
    }
}


inline int blockid(int globalid,  int nb)
{
    return  globalid/nb;
}
inline int localid(int globalid,  int nb)
{
    return  globalid %nb;
}
inline globalid(int localid, int blockid, int nb)
{
    return blockid*nb+localid;
}
inline procid(int blockid, int np)
{
    return blockid%np;
}
inline proclocalid(int blockid, int np)
{
    return blockid/np;
}
inline globalblockid(int localid, int procid, int np)
{
    return localid*np+procid;
}

int pcolid(int i, PPARMS parms, PCONTROLS  controls)
{
    int bid = blockid(i, parms->nb);
    return procid(bid, parms->npcol);
}

    
int have_current_col(int i, PPARMS parms, PCONTROLS  controls)
{
    int bid = blockid(i, parms->nb);
    //    fprintf(stderr,"Procid= %d %d, i, nb, bid, npcol, procid = %d %d %d %d %d\n",
    //	    MP_myprocid(),  controls->rank_in_row, i, parms->nb, bid,
    //	    parms->npcol, procid(bid, parms->npcol));
    return (controls->rank_in_row==procid(bid, parms->npcol));
}

int first_row(int i, PPARMS parms, PCONTROLS  controls)
{
    int bid = blockid(i, parms->nb);
    int currentpid = procid(bid, parms->nprow);
    int first = parms->nb * proclocalid(bid, parms->nprow);
    if (currentpid == controls->rank_in_col){
	first += localid(i, parms->nb);
    }else if (currentpid > controls->rank_in_col){
	first += parms->nb;
    }
    return first;
}
int local_colid(int i, PPARMS parms, PCONTROLS  controls)
{
    int bid = blockid(i, parms->nb);
    int lid = localid(i, parms->nb);
    int lbid = proclocalid(bid, parms->npcol);
    return lid + lbid*parms->nb;
}

int global_rowid(int i, PPARMS parms, PCONTROLS  controls)
{
    int localbid = blockid(i, parms->nb);
    int globalbid = globalblockid(localbid,
				  controls->rank_in_col,
				  parms->nprow);
    return globalid(localid(i, parms->nb), globalbid, parms->nb);
}


int global_colid(int i, PPARMS parms, PCONTROLS  controls)
{
    int localbid = blockid(i, parms->nb);
    int globalbid = globalblockid(localbid,
				  controls->rank_in_row,
				  parms->npcol);
    return globalid(localid(i, parms->nb), globalbid, parms->nb);
}

void convert_global_index_to_local_rows(int gindex, int* pproc, int *plid,
				       PPARMS parms, PCONTROLS controls)
{
    int bid = blockid(gindex, parms->nb);
    int lid = localid(gindex, parms->nb);
    int lbid = proclocalid(bid, parms->nprow);
    *plid= lid + lbid*parms->nb;
    *pproc = procid(bid, parms->nprow);
    
}

void convert_global_index_to_local_cols(int gindex, int* pproc, int *plid,
				       PPARMS parms, PCONTROLS controls)
{
    int bid = blockid(gindex, parms->nb);
    int lid = localid(gindex, parms->nb);
    int lbid = proclocalid(bid, parms->npcol);
    *plid= lid + lbid*parms->nb;
    *pproc = procid(bid, parms->npcol);
    
}

void convert_global_col_range_to_local_range(int c1, int c2,
					     int* lc1, int *lc2,
					     PPARMS parms, PCONTROLS controls)
{
    // c2 must be the real index of the last col, not that +1
    int b1 = blockid(c1, parms->nb);
    int b2 = blockid(c2, parms->nb);
    int myid = controls->rank_in_row;
    int ncols = parms-> npcol;
    if ( (b1 % ncols) == myid){
	*lc1 = localid(c1, parms->nb)+ proclocalid(b1, parms->npcol)*parms->nb;
    }else if ( (b1 % ncols) > myid){
	*lc1 = (proclocalid(b1, parms->npcol)+1)*parms->nb;
    }else{
	*lc1 = (proclocalid(b1, parms->npcol))*parms->nb;
    }
    if ( (b2 % ncols) == myid){
	*lc2 = localid(c2, parms->nb)+ proclocalid(b2, parms->npcol)*parms->nb;
    }else if ( (b2 % ncols) > myid){
	*lc2 = (proclocalid(b2, parms->npcol)+1)*parms->nb -1;
    }else{
	*lc2 = proclocalid(b2, parms->npcol)*parms->nb -1;
    }
}

void swapvoidptr(void **a, void **b)
{
    void * tmp;
    tmp = *a;
    *a = *b ;
    *b = tmp;
}
void swapdoubleptr(double **a, double **b)
{
    double * tmp;
    tmp = *a;
    *a = *b ;
    *b = tmp;
}
