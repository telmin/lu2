#!/bin/csh -f
#
# mympirun.csh
#
# script to run openmpi or mvapich1 in mgv
#
#
# usage: mympirun2.csh np name_of_hostfile  command and options
#
# example
#
#  mympirun2.csh 4 hostfile45  lu2_mpi_gdr -p 4 -n 16384
#
# JM 2010/5/22
#
#set mpitype = openmpi
set mpitype = mvapich1
set np = $1
set hostfile = $2
shift
shift
foreach host ( `cat $hostfile` )
  echo $host
#  rsh  $host 'rm /mnt/huge/a* ' </dev/null 
#  rsh $host ls -al /etc/hosts >/tmp/hosts-${host}.log
   ssh $host killall $1
   ssh $host uname -a
#   ssh $host singclock.csh 24 </dev/null 
end
sleep 2

if ( $mpitype == openmpi ) then
echo  "mpirun     -np $np   -hostfile  $hostfile  $* "
    mpirun     -np $np   -hostfile  $hostfile  $* 
endif
if ( $mpitype == mvapich1 ) then
/usr/mpi/gcc/mvapich-1.2.0/bin/mpirun_rsh -legacy -hostfile  $hostfile -np $np VIADEV_USE_AFFINITY=0 $*
endif
