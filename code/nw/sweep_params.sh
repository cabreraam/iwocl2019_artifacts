#!/bin/bash

# This script takes sweeps BSIZE and PAR of the nw kernel from the rodinia suite.
# Author: Anthony Cabrera
# Email: acabrera@wustl.edu

#BSIZE=( 256 512 1024 2048 4096 8192 )
#BSIZE=( 2048 4096 8192 )
#PAR=( 8 16 32 64 )
BSIZE=( 1024 )
PAR=( 32 )
DATA_DIR="sweep_data"

echo "Starting script!"
for bsize in ${BSIZE[@]};
do
    for par in ${PAR[@]};
    do
        if [[ "$bsize" -eq 256 || "$bsize" -eq 512 || "$bsize" -eq 1024 ]] \
            && [[ "$par" -eq 64 ]]
        then
            echo "NOPE"
        else
            echo "BSIZE: ${bsize}, PAR: ${par}"
			FILENAME="${DATA_DIR}/data_${bsize}_${par}.txt"
			# run make on nw_harp_test with new current bsize and par
			make verify ALTERA=1 BSIZE=${bsize} PAR=${par} 2> make.err
			# copy the right bsize_par.aocx to nw_kernel_v5.aocx
			cp ./bin_fpga/nw_kernel_v5_${bsize}_${par}.aocx ./bin_fpga/nw_kernel_v5.aocx
			#for i in `seq 0 99`; 
			for i in `seq 0 99`; 
			do
            	echo "BSIZE: ${bsize}, PAR: ${par}, run ${i}" >> ${FILENAME}
					./run v5 >> ${FILENAME} 2>&1
					#echo "hi" >> ${FILENAME} 2>&1
				echo "" >> ${FILENAME}
			done
			echo ""
        fi
    done
done
