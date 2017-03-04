lu2_mpi と lu2_mpi_gdr インストールとテスト実行メモ 

Time-stamp: <11/06/29 19:29:35 makino>

Ver 1.0 J. Makino  2010/5/30

= はじめに

このメモでは、

* lu2_mpi と lu2_mpi_gdr のコンパイル方法
* 実行方法、実行結果チェック方法
* mgv での種々のテスト方法

について簡単に説明します。

= lu2_mpi と lu2_mpi_gdr のコンパイル方法

まず、公開版のソースを
grape.mtk.nao.ac.jp/pub/people/makino/softwares/lu/index.html
からダウンロードして、

  tar xvzf lu.tgz

で展開して下さい。これはカレントディレクトリにそのまま展開するので、
あらかじめ適当なディレクトリを作って移動しておいて下さい(これはそのう
ち変更するかもしれません)

コンパイルするためにには、いくつかの設定をMakefile に書く必要があります。
これは configure でできるべきですが、牧野が autoconf の使いかたをまだ会
得していないので、、、設定が必要なのは

CC =
MPICC =
CCFLAGS =
CBLASINC =
GOTOBLASLIB =
GDRLIBS =

です。ちなみに、 mgv での現行の設定は

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

となっていて、これは Makefile ではなく Makefile.mgv に書かれています。
なので、 mgv でコンパイルするのであれば

 make -f Makefile.mgv

で、lu2, lu2_gdr, lu2_mpi, lu2_mpi_gdr のすべてがコンパイル・リンクさ
れるはずです。Makefile のほうは、 GRAPE-DR のことは知らないものになっ
ています。

RHEL6 での設定は Makefile.RHEL6 にあります。


= 実行方法、実行結果チェック方法

lu2 は豪快にスタックを使うので、MPI の場合は自分が使うシェルの初期設定
で limit コマンドでスタックを増やすようにしておいて下さい。128MB くらい
あれば多分大丈夫です。csh の場合、 .cshrc に

   limit stacksize 128M

のはずです。sh, bash は知らないので、調べて下さい。なお、インタラクティ
ブシェルかどうかを判断してそうでなければ余計なことはしないようなこった
設定の .cshrc を使っている場合には、ちゃんとインタラクティブでない時に
も limit が実行されていることを確認して下さい。mpirun を起動するノード
で、例えば 

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

という感じで、 rsh (mpi が rsh 使う設定なら)で limit を実行した時の
stacksize をみて下さい。デフォルト(8MBくらい？)のままだと、途中でエラー
になって止まります。

実行方法はどのような MPICC コマンドを使ったかによるわけですが、上の
mgv 環境では、 mympirun2.csh を使うのが簡単です。これは

 mympirun2.csh np name_of_hostfile  command and options

という方法で実行します。例えば

 mympirun2.csh 4 allhosts  lu2_mpi_gdr -p 4 -n 16384 -g

とします。 lu2_mpi には多数のオプションがありますが、 -h オプションでそ
の一覧を見ることができます。以下のベンチマーク例で使っているオプション
について説明します。


 #!/bin/csh -f
 foreach n (16384 24576 32768 40960 49152 57344 65536 73728 81920 86016)
      mympirun2.csh 64 allhosts lu2_mpiuc_gdr -p 8 -q 8 -n $n -B0 -g   -T8 -N1
      sleep 200
 end

あ、まず、このベンチマークでは  lu2_mpi_gdr ではなく lu2_mpiuc_gdr  を
使っています。これは、1カードでは動作する縦通信並列版です。これは、

 make -f Makefile.mgv lu2_mpiuc_gdr

でコンパイルできるはずです。

 -p 8 横方向のプロセッサ数です。この場合8です。
 -q 8 縦方向のプロセッサ数です。この場合8です。

 これらの指定によりプロセッサ総数は 64 となります。これは、最初の引数
 と一致してなければなりません。これは計算できたほうが賢いですね、、、
 
 
 -n $n 行列サイズです。上の例ではいくつかの行列サイズで順次計算します。
 -B0 複数カードを認識しているシステムの場合にカード0 が最初に使うカー
     ドであることを指定します。
 -g  Hugetlbfs を使うことを指定します。
 -T8 詳細なタイミング情報を出すノード数を指定します。 0 では全ノードの
     情報をだします。
 -N1 複数カードを認識しているシステムの場合に使う枚数を指定します。
     なお、3枚以上では動作しないと思います。これは lu2 の制限ではなく
     て現在のライブラリの制限です。

結果チェックですが、現在はちゃんと HPL 風の出力(そのまま Top500 に登録できる)
がでます。

  Error = 5.295703e-08

といった感じの出力が最後のほうにでるので、この値が 1e-7 以下であれば多
分ちゃんと計算できています。

==  mgv での種々のテスト方法

基本的に、上のテストスクリプトのようなものを作って実行して下さい。ハー
ドウェアの状態をチェックするには、 singletest.csh が有用です。これは、
指定した hostfile に書いてある全ノードで、lu2_mpi(uc)_gdr を単一ノード
実行します。例えば

 csh -f singletest.csh allhosts 40960

で、全ノードで 40k サイズの計算を5回やって、結果を
/tmp/lu2mpi-hostname.log というファイルに作ります。この結果から

   grep Err /tmp/lu2mpi*.log

として、答に再現性がなかったり、他のと答が違うノードは不良です。

= ノードの状態チェック

これは lu2 とは直接関係ないので、そのうちに別の場所に整理して書きます
が、とりあえず。


/usr2/makino/src/pccluster/gdrcluster にある

checkmachines.rb -r 4..12

で、そこに書いてある全ノード (現在、 04-01 から 12-12 まで)で
testdgemm2 を実行し、結果を /tmp/singinit-*.log に書きます。これを実行
するノードからは mgv ノードに rsh を掛けることができる必要があります。
一部のラックだけチェックするには、例えば

checkmachines.rb -r 4..4

でラック4だけになります。これの実行のためにはいくつかの環境変数の設定
が必要です。多分

ACSROOT=/usr2/makino/papers/acs

だけで動くはずです。これで×なら

ACSBIN=/home/makino/bin
ACSSCRIPTS=/usr2/makino/papers/acs/bin
ACSLIBS=/usr2/makino/papers/acs/lib

を設定してみて下さい。

これを実行したノードで

csh -f checkgdr.csh name_of_hostfile

を実行すると、 hostfile に書いてあるノードについて、実行結果がもっとも
らしいかどうか判断して間違っていたら

  mgv04-04 broken...

という感じのメッセージをだします。

また、

csh -f fixgdr.csh name_of_hostfile

で、 singinit.rb をエラーがでたと判定された各ノードで実行してなんとか
しようとします。

また、

makemachinefile.rb

で、ちゃんと動いたノードのリストを標準出力に出すので。これをリダイレク
トすることで hostfile を作ることができます。これで作ったものを、さらに
上の  singletest.csh でチェックすることで一応動くはずの環境を作ること
ができます。

