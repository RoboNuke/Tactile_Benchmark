#!/bin/bash

#!/bin/bash
#SBATCH -J batchTest			# name of job
##SBATCH -A mySponsoredAccount	# name of my sponsored account, e.g. class or research group, NOT ONID!
#SBATCH -p preempt				# name of partition or queue
#SBATCH --time=0-5:00:00        # time limit on job: 2 days, 12 hours, 30 minutes (default 12 hours)
#SBATCH -N 1                    # number of nodes (default 1)
#SBATCH --gres=gpu:1            # number of GPUs to request (default 0)
#SBATCH --mem=32G               # request 10 gigabytes memory (per node, default depends on node)
##SBATCH --constraint=ib        # request node with infiniband
##SBATCH --constraint=avx512    # request node with AVX512 instruction set
##SBATCH --constraint=a40       # request node with A40 GPU


# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load cuda/10.1
eval "$(conda activate mani)"

#hol=nvidia-smi --query-gpu=memory.free --format=csv,noheader
# run my job (e.g. matlab, python)
bash /nfs/stak/users/brownhun/hpc-share/Tactile_Benchmark/learning/HPC/10_launch_tmux.bash 1 1 "FragilePegInsert-v1"  \
    state_dict 100 testerMcTesty 150 500000 pd_joint_delta_pos \
    normalized_dense FFN