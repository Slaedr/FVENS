#! /bin/bash
#PBS -l nodes=1:ppn=16
#PBS -l walltime=01:00:00
#PBS -A rck-371-aa
#PBS -o supersonicvortex-impl.log
#PBS -e supersonicvortex-impl.err
#PBS -N svortex

# to be executed from Euler1d-acc/runs/ or equivalent

# ppn=16:sandybridge will always be run on a SW2 sandy bridge node, according to http://www.hpc.mcgill.ca/index.php/starthere/81-doc-pages/91-guillimin-job-submit

threadseq='1 2 4 8 10 16'

cd $PBS_O_WORKDIR
module load intel/2017.00

for i in ${threadseq}; do

	export OMP_NUM_THREADS=$i
	time ../build/steadyfvensi ../testcases/supersonic-vortex/implicit.control

done
