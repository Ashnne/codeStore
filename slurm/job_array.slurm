#! /bin/bash

#SBATCH --job-name=recon
#SBATCH -p Low
#SBATCH --nodes=1                  # 节点，相当于申请几台计算机
#SBATCH --cpus-per-task=64        # 每个进程的cpu分配数量
#SBATCH --tasks-per-node=1        # 每个node上有几个进程在跑
#SBATCH --gres=gpu:4                # 每个node要几个gpu，一般看一个进程要多少gpu
#SBATCH --mem=256GB                 # Memory per node
#SBATCH --time=96:00:00             # Time limit (96 hours)
#SBATCH --output=slurm_log/slurm_%A_%a.out   # %A-job index %a-array index
#SBATCH --error=slurm_log/slurm_%A_%a.out    # %A-job index %a-array index
#SBATCH --array=0-15

num=$1

export NCCL_DEBUG=INFO                      # Optional: for debugging NCCL issues
export MASTER_ADDR=$(scontrol show hostname | head -n 1)  # Get the master node's hostname
export MASTER_PORT=$((29501+num))                    # Port for distributed communication

# 这个可以使用conda环境
# 这个需要改成自己的conda的activate命令位置
source /public/home/group_yangych/qyzheng/anaconda3/bin/activate infoaug


# SLURM_ARRAY_TASK_ID和SLURM_JOB_ID两个环境变量分别是array的index和job id
# 可以用于区分不用的array task做不同的任务
index=$((SLURM_ARRAY_TASK_ID+48))

echo "Job array task ID: $SLURM_ARRAY_TASK_ID"
echo "run the index: $index"

cd $index
work_path=$(pwd)

cd ../../..

# 告诉mian函数你是那个index
srun python main.py \
        --index $index --cuda 0