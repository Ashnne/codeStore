# slurm简易教程
slurm集群一般使用srun进行交互调试，
调试结束后使用sbatch提交写好的脚本进行训练，
以便于资源的高效应用

在SH-HKU集群中，可以在网页使用ui注册容器申请资源来调试，但在大规模使用资源的时候最好使用sbatch提交脚本，因为1） 这样可以更高效的使用资源，避免空挂 2） 批量操作，不需要反复使用ssh连接控制容器，可以随时查看输出，减少人力成本 3） 使用slurm的命令比起网页ui注册，可以加快io速度

## slurm脚本的一些基础设置

```
#! /bin/bash


# 一些关于申请资源配置文件
#SBATCH --job-name=recon              # 任务名
#SBATCH -p gpu                        # 资源的队列 SH-HKU集群中，有gpux gpu Low High cpu几个队列，其中使用4/8 gpus的任务将自动归类到gpux队列中，不使用gpu的任务将自动归类到cpu队列中
#SBATCH --nodes=1                     # 申请节点的数量
#SBATCH --cpus-per-task=16            # 每个节点上的cpu 
# TODO: 把task搞清楚怎么用
#SBATCH --tasks-per-node=1            # 每个节点上的task GLOBAL_RANK 和 LOCAL_RANK
#SBATCH --gres=gpu:1                  # 每个节点上的GPU
#SBATCH --mem=256GB                   # 每个节点的内存
#SBATCH --time=96:00:00               # 时间临界
#SBATCH --output=slurm_log/test1.out # out 输出存放文件 # 以运行的位置的相对位置 # path=${pwd}/${output}
#SBATCH --error=slurm_log/test1.err  # err 报错存放文件  


num=$1
export NCCL_DEBUG=INFO                      # Optional: for debugging NCCL issues
export MASTER_ADDR=$(scontrol show hostname | head -n 1)  # Get the master node's hostname
export MASTER_PORT=$((29501+num))                    # Port for distributed communication # 要错开

# conda activate infoaug
source /public/home/group_yangych/qyzheng/anaconda3/bin/activate infoaug

# 工作目录就是运行目录
# pwd = pwd( before)


# 工作的操作
```

## slurm的使用例子

在[范例脚本](slurm.slurm)中，我们想要申请1个节点，4张GPU，64个CPU，256G内存的资源。

同时，我们想要运行[目标文件](slurm.py)，在一张GPU上我们想要同时进行两个目标文件的运行，通过参数index和cuda来确定自己使用的显卡和编号。

为了让[目标文件](slurm.py)能够得到自己的编号和显卡，同时也让该节点可以同时运行多个进程，我们使用[脚本](run.sh)来进行控制，在[范例脚本](slurm.slurm)中，通过后台运行[脚本](run.sh)的方式来进行多进程操作，用wait命令来让slurm任务等待后台进程结束后退出。

因此在这个例子中，通过sbatch slurm.slurm命令：

1. 会向集群提交1个节点，4张GPU，64个CPU，256G内存的资源申请。
2. 申请到资源后，激活conda环境，并依次在后台挂起run.sh脚本指令，并在wait命令处等待后台进程结束
3. run.sh的命令在后台执行main.py脚本，并提供对应的index和cuda参数
4. main.py脚本全部执行完毕后，wait命令确定后台进程执行完，结束任务，释放资源