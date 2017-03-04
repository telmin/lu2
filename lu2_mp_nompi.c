//
// lu2_mp_nompi.c
//
#define HUGEPAGE
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>

#include <emmintrin.h>

static int local_proc_id = 0;
static int total_proc_count = 1;

    

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
    
}


void MP_initialize(int * argc,  char *** argv)
{

}


void  MP_sync()
{
}


void MP_end()
{

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

