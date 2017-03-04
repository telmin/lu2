#!/bin/csh -f
#
# mympirun.csh
#
# script to run openmpi or mvapich1 in mgv
#
#
# usage: mympirun2b.csh  np name_of_hostfile  command and options
#
# example
#
#  mympirun2b.csh  4 hostfile45  lu2_mpi_gdr -p 4 -n 16384
#
# JM 2010/5/22
#
#set mpitype = openmpi133
set mpitype = mvapich1
set np = $1
set hostfile = $2
shift
shift
foreach host ( `cat $hostfile` )
  echo $host
  rsh  $host 'rm /mnt/huge/a* ' </dev/null &
end
sleep 5

if ( $mpitype == openmpi133 ) then
/usr2/makino/PDS/openmpi133/bin/mpirun \
   --mca btl openib,sm,self\
   --mca mpi_leave_pinned 1 \
  --mca btl_openib_ib_timeout 30 \
   --mca btl_openib_use_srq 1 \
   --mca btl_openib_flags 6 \
   --mca btl_openib_srq_rd_max 100000 \
   --mca  pls_rsh_agent "/usr/bin/rsh" -hostfile $hostfile \
    -np $np $*
#   --mca btl_openib_max_rdma_size 16777216 \
endif
if ( $mpitype == mvapich1 ) then
/usr/mpi/gcc/mvapich-1.1.0/bin/mpirun_rsh -rsh  -hostfile  $hostfile -np $np VIADEV_USE_AFFINITY=0 $*
/usr/mpi/gcc/mvapich-1.1.0/bin/mpirun_rsh -rsh  -hostfile  $hostfile -np $np VIADEV_USE_AFFINITY=0 $*
/usr/mpi/gcc/mvapich-1.1.0/bin/mpirun_rsh -rsh  -hostfile  $hostfile -np $np VIADEV_USE_AFFINITY=0 $*
/usr/mpi/gcc/mvapich-1.1.0/bin/mpirun_rsh -rsh  -hostfile  $hostfile -np $np VIADEV_USE_AFFINITY=0 $*
/usr/mpi/gcc/mvapich-1.1.0/bin/mpirun_rsh -rsh  -hostfile  $hostfile -np $np VIADEV_USE_AFFINITY=0 $*
endif

