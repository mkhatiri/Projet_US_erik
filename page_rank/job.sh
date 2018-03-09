#! /bin/bash




#NPS -N mkhatiri

#PBS -q copperhead


#PBS -l walltime=00:09:00

#PBS -l nodes=1:ppn=1:gpus=2,mem=16GB




#nvcc --version


module load cuda/8.0
module load intel/14.0.3
module load gcc/5.3.0
module load cudnn/6.0-cuda8

export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.5/lib64:$LD_LIBRARY_PATH

echo "exec"
cd /users/mkhatiri/ShMemGraph/src/page_rank

#lspci -t

#nvidia-smi topo -m

#nvidia-smi -a

#for i in $(ls /users/mkhatiri/PROJETCOMPLET/SNAP/*.bin) 
#do
#NBBLOCK=512 NBTHREAD=512 BLKSIZE=128 ./new-cuda-adaptative-2gpus $i 10;
#VAL=0 NBBLOCK=400 NBTHREAD=1024 BLKSIZE=1024 ./new-cuda-adaptative-2gpus $i;
#NBSTREAM=8 NBBLOCK=64 ./new-cuda-pr-2gpus $i 10;
#./cuda-pr $i 10;
#NBSTREAM=4 NBBLOCK=32 ./new-cuda-pr $i;
#done


#NBBLOCK=32 NBTHREAD=1024 BLKSIZE=2048 ./new-cuda-adaptative-2gpus  /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin


#for nb in 8 16 32 64 128
#do
#for nt in 32 64 128 512 
#do
#for bls in 2048 4096
#do
#echo $nb " " $nt " " $bls "\n"
#NBBLOCK=$nb NBTHREAD=$nt BLKSIZE=$bls ./new-cuda-adaptative-2gpus   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 5
#done
#done
#done

#NBBLOCK=4 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#NBBLOCK=64 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#NBBLOCK=128 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#NBBLOCK=256 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 

######################

#NBBLOCK=32 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative-2gpus   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#NBBLOCK=64 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative-2gpus   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#NBBLOCK=128 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative-2gpus   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#NBBLOCK=32 NBTHREAD=128 BLKSIZE=1024 ./new-cuda-adaptative-2gpus   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 

#NBBLOCK=8 NBTHREAD=128 BLKSIZE=1024 ./new-cuda-adaptative   /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 

#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin


#NBSTREAM=8 NBBLOCK=64 ./new-cuda-pr-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin
#NBSTREAM=4 NBBLOCK=32 ./new-cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin

#NBBLOCK=8 NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin 
#NBBLOCK=1 NBTHREAD=128 BLKSIZE=1024 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin 
#NBBLOCK=16 NBTHREAD=512 BLKSIZE=1024 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 


VAL=1 NBBLOCK=2  NBTHREAD=512 BLKSIZE=512  ./new-cuda-adaptative-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin 
#####
#VAL=0 NBBLOCK=250  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin 
#./cuda-pr-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin 
 
#NBSTREAM=4 NBBLOCK=32 ./new-cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin 
#NBSTREAM=8 NBBLOCK=64 ./new-cuda-pr-2gpus /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin 

####
#VAL=1 NBBLOCK=500  NBTHREAD=256 BLKSIZE=1024 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin
#VAL=1 NBBLOCK=64  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0312.bin
#VAL=1 NBBLOCK=64  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0505.bin
#VAL=1 NBBLOCK=64  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/amazon0601.bin
#VAL=1 NBBLOCK=64  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/roadNet-CA.bin
#VAL=1 NBBLOCK=64  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/roadNet-PA.bin
#VAL=1 NBBLOCK=64  NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/roadNet-TX.bin
#VAL=1 NBBLOCK=64 NBTHREAD=128 BLKSIZE=128 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/as-Skitter.bin
#VAL=1 NBBLOCK=512 NBTHREAD=256 BLKSIZE=2048 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/com-orkut.ungraph.bin 


#VAL=1 NBBLOCK=200  NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/ca-AstroPh.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/ca-AstroPh.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/as-735.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/as-735.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/ca-HepTh.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/ca-HepTh.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella04.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella04.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella05.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella05.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella06.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella06.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella08.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella08.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella09.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella09.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella24.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella24.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella30.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/p2p-Gnutella30.bin
#VAL=0 NBBLOCK=500 NBTHREAD=256 BLKSIZE=256 ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/wiki-Vote.bin
#./cuda-pr /users/mkhatiri/PROJETCOMPLET/SNAP/wiki-Vote.bin
#
#
#
#
#
