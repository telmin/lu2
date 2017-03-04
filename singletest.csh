#!/bin/csh -f
#
# shingletest.csh
#
# script to test each nodes in a host file
#
set hostfile = $1
set n = $2
foreach host ( `cat $hostfile` )
  echo -n $host
  echo $host > .myhostfile.${host}
  mympirun2.csh 1 .myhostfile.${host}  lu2_mpi_gdr -p 1 -q 1 -n $n >& /tmp/lu2mpi-${host}.log &
end
