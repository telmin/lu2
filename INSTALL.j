lu2_mpi �� lu2_mpi_gdr ���󥹥ȡ���ȥƥ��ȼ¹ԥ�� 

Time-stamp: <11/06/29 19:29:35 makino>

Ver 1.0 J. Makino  2010/5/30

= �Ϥ����

���Υ��Ǥϡ�

* lu2_mpi �� lu2_mpi_gdr �Υ���ѥ�����ˡ
* �¹���ˡ���¹Է�̥����å���ˡ
* mgv �Ǥμ�Υƥ�����ˡ

�ˤĤ��ƴ�ñ���������ޤ���

= lu2_mpi �� lu2_mpi_gdr �Υ���ѥ�����ˡ

�ޤ��������ǤΥ�������
grape.mtk.nao.ac.jp/pub/people/makino/softwares/lu/index.html
�����������ɤ��ơ�

  tar xvzf lu.tgz

��Ÿ�����Ʋ�����������ϥ����ȥǥ��쥯�ȥ�ˤ��Τޤ�Ÿ������Τǡ�
���餫����Ŭ���ʥǥ��쥯�ȥ���äư�ư���Ƥ����Ʋ�����(����Ϥ��Τ�
���ѹ����뤫�⤷��ޤ���)

����ѥ��뤹�뤿��ˤˤϡ������Ĥ��������Makefile �˽�ɬ�פ�����ޤ���
����� configure �ǤǤ���٤��Ǥ�������� autoconf �λȤ�������ޤ���
�����Ƥ��ʤ��Τǡ��������꤬ɬ�פʤΤ�

CC =
MPICC =
CCFLAGS =
CBLASINC =
GOTOBLASLIB =
GDRLIBS =

�Ǥ������ʤߤˡ� mgv �Ǥθ��Ԥ������

CC = gcc
MPICC = /usr/mpi/gcc/mvapich-1.1.0/bin/mpicc
CCFLAGS = -O2 -I.  -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE \
   -D_FILE_OFFSET_BITS=64  -DTIMETEST  
CBLASDIR = /usr2/makino/PDS/CBLAS
CBLASINC = -I$(CBLASDIR)/src
GOTOBLASDIR = /usr2/makino/PDS/GotoBLAS2test
GOTOBLASLIB = -L$(GOTOBLASDIR)  -lgoto2 -lgfortran
GDRBASELIB = -L/usr2/makino/src/gdrlib/gdrtb3/singlibMC_tb4 -L/usr2/makino/src/gdrlib/gdrtb3/lib -lsing -lhib 
GDRDGEMMLIB = /usr2/makino/src/gdrlib/gdrtb3/singlibMC_tb4/libdgemm2-tst/libgdrdgemm-ovl.a
GDRLIBS = $(GDRDGEMMLIB) $(GDRBASELIB)

�ȤʤäƤ��ơ������ Makefile �ǤϤʤ� Makefile.mgv �˽񤫤�Ƥ��ޤ���
�ʤΤǡ� mgv �ǥ���ѥ��뤹��ΤǤ����

 make -f Makefile.mgv

�ǡ�lu2, lu2_gdr, lu2_mpi, lu2_mpi_gdr �Τ��٤Ƥ�����ѥ��롦��󥯤�
���Ϥ��Ǥ���Makefile �Τۤ��ϡ� GRAPE-DR �Τ��Ȥ��Τ�ʤ���Τˤʤ�
�Ƥ��ޤ���

RHEL6 �Ǥ������ Makefile.RHEL6 �ˤ���ޤ���


= �¹���ˡ���¹Է�̥����å���ˡ

lu2 �Ϲ���˥����å���Ȥ��Τǡ�MPI �ξ��ϼ�ʬ���Ȥ�������ν������
�� limit ���ޥ�ɤǥ����å������䤹�褦�ˤ��Ƥ����Ʋ�������128MB ���餤
�����¿ʬ����פǤ���csh �ξ�硢 .cshrc ��

   limit stacksize 128M

�ΤϤ��Ǥ���sh, bash ���Τ�ʤ��Τǡ�Ĵ�٤Ʋ��������ʤ������󥿥饯�ƥ�
�֥����뤫�ɤ�����Ƚ�Ǥ��Ƥ����Ǥʤ����;�פʤ��ȤϤ��ʤ��褦�ʤ��ä�
����� .cshrc ��ȤäƤ�����ˤϡ������ȥ��󥿥饯�ƥ��֤Ǥʤ�����
�� limit ���¹Ԥ���Ƥ��뤳�Ȥ��ǧ���Ʋ�������mpirun ��ư����Ρ���
�ǡ��㤨�� 

 $ rsh mgv07-01 limit
 cputime      unlimited
 filesize     unlimited
 datasize     unlimited
 stacksize    131072 kbytes
 coredumpsize 0 kbytes
 memoryuse    unlimited
 vmemoryuse   unlimited
 descriptors  1024 
 memorylocked unlimited
 maxproc      159744 

�Ȥ��������ǡ� rsh (mpi �� rsh �Ȥ�����ʤ�)�� limit ��¹Ԥ�������
stacksize ��ߤƲ��������ǥե����(8MB���餤��)�Τޤޤ��ȡ�����ǥ��顼
�ˤʤäƻߤޤ�ޤ���

�¹���ˡ�ϤɤΤ褦�� MPICC ���ޥ�ɤ�Ȥä����ˤ��櫓�Ǥ��������
mgv �Ķ��Ǥϡ� mympirun2.csh ��Ȥ��Τ���ñ�Ǥ��������

 mympirun2.csh np name_of_hostfile  command and options

�Ȥ�����ˡ�Ǽ¹Ԥ��ޤ����㤨��

 mympirun2.csh 4 allhosts  lu2_mpi_gdr -p 4 -n 16384 -g

�Ȥ��ޤ��� lu2_mpi �ˤ�¿���Υ��ץ���󤬤���ޤ����� -h ���ץ����Ǥ�
�ΰ����򸫤뤳�Ȥ��Ǥ��ޤ����ʲ��Υ٥���ޡ�����ǻȤäƤ��륪�ץ����
�ˤĤ����������ޤ���


 #!/bin/csh -f
 foreach n (16384 24576 32768 40960 49152 57344 65536 73728 81920 86016)
      mympirun2.csh 64 allhosts lu2_mpiuc_gdr -p 8 -q 8 -n $n -B0 -g   -T8 -N1
      sleep 200
 end

�����ޤ������Υ٥���ޡ����Ǥ�  lu2_mpi_gdr �ǤϤʤ� lu2_mpiuc_gdr  ��
�ȤäƤ��ޤ�������ϡ�1�����ɤǤ�ư�����̿������ǤǤ�������ϡ�

 make -f Makefile.mgv lu2_mpiuc_gdr

�ǥ���ѥ���Ǥ���Ϥ��Ǥ���

 -p 8 �������Υץ��å����Ǥ������ξ��8�Ǥ���
 -q 8 �������Υץ��å����Ǥ������ξ��8�Ǥ���

 �����λ���ˤ��ץ��å������ 64 �Ȥʤ�ޤ�������ϡ��ǽ�ΰ���
 �Ȱ��פ��Ƥʤ���Фʤ�ޤ��󡣤���Ϸ׻��Ǥ����ۤ��������Ǥ��͡�����
 
 
 -n $n ���󥵥����Ǥ��������ǤϤ����Ĥ��ι��󥵥����ǽ缡�׻����ޤ���
 -B0 ʣ�������ɤ�ǧ�����Ƥ��륷���ƥ�ξ��˥�����0 ���ǽ�˻Ȥ�����
     �ɤǤ��뤳�Ȥ���ꤷ�ޤ���
 -g  Hugetlbfs ��Ȥ����Ȥ���ꤷ�ޤ���
 -T8 �ܺ٤ʥ����ߥ󥰾����Ф��Ρ��ɿ�����ꤷ�ޤ��� 0 �Ǥ����Ρ��ɤ�
     ���������ޤ���
 -N1 ʣ�������ɤ�ǧ�����Ƥ��륷���ƥ�ξ��˻Ȥ��������ꤷ�ޤ���
     �ʤ���3��ʾ�Ǥ�ư��ʤ��Ȼפ��ޤ�������� lu2 �����¤ǤϤʤ�
     �Ƹ��ߤΥ饤�֥������¤Ǥ���

��̥����å��Ǥ��������ߤϤ����� HPL ���ν���(���Τޤ� Top500 ����Ͽ�Ǥ���)
���Ǥޤ���

  Error = 5.295703e-08

�Ȥ��ä������ν��Ϥ��Ǹ�Τۤ��ˤǤ�Τǡ������ͤ� 1e-7 �ʲ��Ǥ����¿
ʬ�����ȷ׻��Ǥ��Ƥ��ޤ���

==  mgv �Ǥμ�Υƥ�����ˡ

����Ū�ˡ���Υƥ��ȥ�����ץȤΤ褦�ʤ�Τ��äƼ¹Ԥ��Ʋ��������ϡ�
�ɥ������ξ��֤�����å�����ˤϡ� singletest.csh ��ͭ�ѤǤ�������ϡ�
���ꤷ�� hostfile �˽񤤤Ƥ������Ρ��ɤǡ�lu2_mpi(uc)_gdr ��ñ��Ρ���
�¹Ԥ��ޤ����㤨��

 csh -f singletest.csh allhosts 40960

�ǡ����Ρ��ɤ� 40k �������η׻���5���äơ���̤�
/tmp/lu2mpi-hostname.log �Ȥ����ե�����˺��ޤ������η�̤���

   grep Err /tmp/lu2mpi*.log

�Ȥ��ơ����˺Ƹ������ʤ��ä��ꡢ¾�Τ������㤦�Ρ��ɤ����ɤǤ���

= �Ρ��ɤξ��֥����å�

����� lu2 �Ȥ�ľ�ܴط��ʤ��Τǡ����Τ������̤ξ����������ƽ񤭤ޤ�
�����Ȥꤢ������


/usr2/makino/src/pccluster/gdrcluster �ˤ���

checkmachines.rb -r 4..12

�ǡ������˽񤤤Ƥ������Ρ��� (���ߡ� 04-01 ���� 12-12 �ޤ�)��
testdgemm2 ��¹Ԥ�����̤� /tmp/singinit-*.log �˽񤭤ޤ��������¹�
����Ρ��ɤ���� mgv �Ρ��ɤ� rsh ��ݤ��뤳�Ȥ��Ǥ���ɬ�פ�����ޤ���
�����Υ�å����������å�����ˤϡ��㤨��

checkmachines.rb -r 4..4

�ǥ�å�4�����ˤʤ�ޤ�������μ¹ԤΤ���ˤϤ����Ĥ��δĶ��ѿ�������
��ɬ�פǤ���¿ʬ

ACSROOT=/usr2/makino/papers/acs

������ư���Ϥ��Ǥ�������ǡߤʤ�

ACSBIN=/home/makino/bin
ACSSCRIPTS=/usr2/makino/papers/acs/bin
ACSLIBS=/usr2/makino/papers/acs/lib

�����ꤷ�ƤߤƲ�������

�����¹Ԥ����Ρ��ɤ�

csh -f checkgdr.csh name_of_hostfile

��¹Ԥ���ȡ� hostfile �˽񤤤Ƥ���Ρ��ɤˤĤ��ơ��¹Է�̤���äȤ�
�餷�����ɤ���Ƚ�Ǥ��ƴְ�äƤ�����

  mgv04-04 broken...

�Ȥ��������Υ�å�����������ޤ���

�ޤ���

csh -f fixgdr.csh name_of_hostfile

�ǡ� singinit.rb �򥨥顼���Ǥ���Ƚ�ꤵ�줿�ƥΡ��ɤǼ¹Ԥ��Ƥʤ�Ȥ�
���褦�Ȥ��ޤ���

�ޤ���

makemachinefile.rb

�ǡ�������ư�����Ρ��ɤΥꥹ�Ȥ�ɸ����Ϥ˽Ф��Τǡ�����������쥯
�Ȥ��뤳�Ȥ� hostfile ���뤳�Ȥ��Ǥ��ޤ�������Ǻ�ä���Τ򡢤����
���  singletest.csh �ǥ����å����뤳�Ȥǰ��ư���Ϥ��δĶ����뤳��
���Ǥ��ޤ���

