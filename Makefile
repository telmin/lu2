CC = gcc
#MPICC = /home/makino/PDS/mpic*/bin/mpicc 
MPICC = mpicc 
CCFLAGS = -O2 -I.  -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 -DTIMETEST -DHUGEPAGE
# -O4 does not work on zel2


CBLASDIR = $(HOME)/PDS/CBLAS
#GOTOBLASDIR = $(HOME)/PDS/GotoBLAS1.26
#GOTOBLASLIB = -L$(GOTOBLASDIR) -lgoto
GOTOBLASDIR = $(HOME)/libs/OpenBLAS
CBLASINC=-I $(GOTOBLASDIR)/include
GOTOBLASLIB = -L$(GOTOBLASDIR)/lib  -lopenblas

include Makefile.common

# local target. Not for use by users
# tranfer files from Makino's local PC to MITAKA GRAPE-DR site
send : $(SRCS)
	putfilemout.csh $(SRCS)
	putfilet.csh $(SRCS)
