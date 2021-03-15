#!/bin/bash
#SBATCH --job-name=CL_DCASE2018_01
#SBATCH --partition=gpu #set to GPU for GPU usage
#SBATCH --nodes=1              # number of nodes
#SBATCH --mem=60GB               # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=64    # number of cores
#SBATCH --time=72:00:00           # HH-MM-SS
#SBATCH --output /home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/DCASE2018/output/DCASE2018_job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error /home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/DCASE2018/output/DCASE2018_tf.%N.%j.err  # filename for STDERR
# to load CUDA module
module load cuda

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/CNNLSTM

# locate to your root directory
cd /home/mkolpe2s/rand/Classic_ML/Proper_method/KNN/DCASE2018
# run the script
python KNN_DCASE2018.py

#Find your batch status in https://wr0.wr.inf.h-brs.de/wr/stat/batch.xhtml

