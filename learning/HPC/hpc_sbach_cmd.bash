#!/bin/bash
#SBATCH --array=1-2             # set up the array
#SBATCH -J Ent001			    # name of job
#SBATCH -A virl-grp	            # name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p eecs2 # dgx2				# name of partition or queue
#SBATCH --time=0-12:00:00        # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
##SBATCH -N 1                   # number of nodes (default 1)
#SBATCH --gres=gpu:1            # number of GPUs to request (default 0)
#SBATCH --mem=16G               # request 10 gigabytes memory (per node, default depends on node)
#SBATCH -c 2                    # number of cores/threads per task (default 1)
#SBATCH -o ../outs/Ent001_%A_%a.out		# name of output file for this submission script
#SBATCH -e ../outs/Ent001_%A_%a.err		# name of error file for this submission script
# load any software environment module required for app (e.g. matlab, gcc, cuda)


module load cuda/10.1
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate mani
#echo $CONDA_DEFAULT_ENV
#echo starting_process
#hol=nvidia-smi --query-gpu=memory.free --format=csv,noheader
# run my job (e.g. matlab, python)
if [ $SLURM_ARRAY_TASK_ID == 1 ]; then
    beg_idx=1
    end_idx=5
else
    beg_idx=6
    end_idx=10
fi
echo $beg_idx $end_idx $SLURM_ARRAY_TASK_ID
bash learning/HPC/hpc_launch.bash $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID $*
#bash learning/HPC/10_launch_tmux.bash $beg_idx $end_idx $*
