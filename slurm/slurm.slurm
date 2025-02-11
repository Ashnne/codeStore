#! /bin/bash

#SBATCH --job-name=recon
#SBATCH -p Low
#SBATCH --nodes=1                  # 节点，相当于申请几台计算机
#SBATCH --cpus-per-task=64        # 每个进程的cpu分配数量
#SBATCH --tasks-per-node=1        # 每个node上有几个进程在跑
#SBATCH --gres=gpu:4                # 每个node要几个gpu，一般看一个进程要多少gpu
#SBATCH --mem=256GB                 # Memory per node
#SBATCH --time=96:00:00             # Time limit (96 hours)
#SBATCH --output=slurm.out
#SBATCH --error=slurm.err

num=$1

export NCCL_DEBUG=INFO                      # Optional: for debugging NCCL issues
export MASTER_ADDR=$(scontrol show hostname | head -n 1)  # Get the master node's hostname
export MASTER_PORT=$((29501+num))                    # Port for distributed communication

# 这个可以使用conda环境
# 这个需要改成自己的conda的activate命令位置
source /public/home/group_yangych/qyzheng/anaconda3/bin/activate infoaug

work_path=$(pwd) # 这个就是使用sbatch的目录

for i in $(seq 0 7)
do
bash run.sh $i &
done

# 等待后台进程全部结束后再退出，可以用一个节点进行多个任务操作
wait