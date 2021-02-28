#!/bin/sh

#SBATCH --job-name=DNA_regression
#SBATCH --output=tensorflow_%j.log
#SBATCH --nodes=1
##SBATCH --ntasks-per-node=12
#SBATCH --partition=debug
##SBATCH --nodelist=node02
##SBATCH --exclusive
export MV2_HOMOGENEOUS_CLUSTER=1
export MV2_IBA_EAGER_THRESHOLD=32K
#--mpi=pmi2
python DNASeq_tensorflow.py

