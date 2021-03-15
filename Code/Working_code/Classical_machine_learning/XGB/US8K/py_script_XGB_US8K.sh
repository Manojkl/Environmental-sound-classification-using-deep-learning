#!/bin/bash
#SBATCH --job-name=CL_US8K_06
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=30GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/mkolpe2s/rand/Classic_ML/Proper_method/XGB/US8K/output/US8K_job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/mkolpe2s/rand/Classic_ML/Proper_method/XGB/US8K/output/US8K_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/CNNLSTM

# locate to your root directory
cd /home/mkolpe2s/rand/Classic_ML/Proper_method/XGB/US8K
# run the script
python XGB_US8K.py

#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml

