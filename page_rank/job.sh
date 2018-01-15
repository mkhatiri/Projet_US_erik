#! /bin/bash




#NPS -N mkhatiri

#PBS -q python

#PBS -l nodes=1:ppn=2:gpus=3  


module load cuda/7.5
module load intel/14.0.3
module load gcc/5.3.0

export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH

echo "exec"
cd /users/mkhatiri/ShMemGraph/src/page_rank

#lspci -t

#nvidia-smi topo -m

#nvidia-smi -a

#for i in $(ls /users/mkhatiri/PROJETCOMPLET/SNAP/*.bin) 
#do
#NBBLOCK=64 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative $i;
#NBSTREAM=4 NBBLOCK=16 ./new-cuda-pr-2gpus $i;
#NBSTREAM=4 NBBLOCK=16 ./new-cuda-pr $i;
#done
#./p 100 10 4 1 0 #





for i in cit-HepPh.bin cit-HepTh.bin email-EuAll.bin p2p-Gnutella05.bin p2p-Gnutella06.bin p2p-Gnutella08.bin p2p-Gnutella09.bin p2p-Gnutella24.bin p2p-Gnutella25.bin p2p-Gnutella30.bin soc-sign-epinions.bin soc-sign-Slashdot081106.bin soc-sign-Slashdot090216.bin soc-sign-Slashdot090221.bin soc-Slashdot0902.bin web-BerkStan.bin web-Google.bin  
do
NBBLOCK=64 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/$i
NBSTREAM=4 NBBLOCK=16 ./new-cuda-pr-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/$i
NBSTREAM=4 NBBLOCK=16 ./new-cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/$i
done





#NBBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin


#echo "64 512 2048"
#BBLOCK=64 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin 
#echo "128 512 2048"
#echo "256 512 2048"
#BBLOCK=256 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin 
#
#echo "512 512 2048"
#BBLOCK=512 NBTHREAD=512 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin 
#
