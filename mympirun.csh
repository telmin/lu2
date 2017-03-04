#!/bin/csh -f
#
# mympirun.csh
#
# script to run openmpi or mvapich1 in mgv
#
#
# usage: mympirun.csh np rackid "list of nodes"  command and options
#
# example
#
#  mympirun.csh 4  "01-01 02  06 07 " lu2_mpi_gdr -p 4 -n 16384
#  mympirun.csh 4  "01-01 02  03-06 07 " lu2_mpi_gdr -p 2 -q 2 -n 16384
#
# JM 2009/11/3
#
#set mpitype = openmpi
set mpitype = mvapich1
set np = $1
shift
set machinefile = ".myhostfile."$$
ruby generate_hostfile.rb $1 > $machinefile
foreach hostname  ( `cat $machinefile` )
#  echo removing hugetlb file on host $hostname
#  rsh  $hostname 'rm /mnt/huge/a*'
#  rsh $hostname singclock.csh 22
#  rsh  $hostname  ~/src/linsol/l*2*tst/testdgemm2 -N2 -g 
end
shift
if ( $mpitype == openmpi) then
echo  "mpirun     -np $np   -hostfile  ${cwd}/$machinefile  $* "
    mpirun     -np $np   -hostfile  ${cwd}/$machinefile  $* 
endif
if ( $mpitype == mvapich1 ) then
#/usr/mpi/gcc/mvapich-1.2.0/bin/mpirun_rsh -rsh  -hostfile  ${cwd}/$machinefile -np $np VIADEV_USE_AFFINITY=0 $*
/usr/mpi/gcc/mvapich-1.2.0/bin/mpirun_rsh -legacy  -hostfile  ${cwd}/$machinefile -np $np VIADEV_USE_AFFINITY=0 $*
endif

