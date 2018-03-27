#! /bin/bash


WorkDir=$1

cd $1

for i in $(ls *.o )
do
	grep ".uncc.edu" $i > $i.data
	grep "Avg"  $i >> $i.data
done 
