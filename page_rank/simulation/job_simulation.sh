#! /bin/bash




#NPS -N mkhatiri

#PBS -q copperhead


#PBS -l walltime=00:45:00

#PBS -l nodes=1:ppn=1:gpus=2,mem=32GB


module load cuda/8.0
module load intel/14.0.3
module load gcc/5.3.0
#module load cudnn/6.0-cuda8

#export PATH=/usr/local/cuda-8.0/bin:$PATH
#export PATH=/usr/local/cuda-8.0/bini/nvprof:$PATH
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH

hostname

uptime

echo "exec"
cd /users/mkhatiri/ShMemGraph/src/page_rank

#lspci -t

#nvidia-smi topo -m

#nvidia-smi -a


Graph_Name=${graph}



./cuda-pr  /users/mkhatiri/PROJETCOMPLET/SNAP/$Graph_Name 10


deviceQuery 1>&2

for shortblksz in 8 16 32 64 128
do	
	for blksz in 128 256 512 1024
	do
	 NBBLOCK=1 SHORTBLKSIZE=$shortblksz  NBTHREAD=$((blksz/4))  BLKSIZE=$blksz ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/$Graph_Name 10
	 NBBLOCK=1 SHORTBLKSIZE=$shortblksz  NBTHREAD=$((blksz/2))  BLKSIZE=$blksz ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/$Graph_Name 10
	 NBBLOCK=1 SHORTBLKSIZE=$shortblksz  NBTHREAD=$blksz  BLKSIZE=$blksz ./new-cuda-adaptative /users/mkhatiri/PROJETCOMPLET/SNAP/$Graph_Name 10
	 done
done



