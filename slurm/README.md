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

### 第一个slurm脚本
在[范例脚本](slurm.slurm)中，我们想要申请1个节点，4张GPU，64个CPU，256G内存的资源。

同时，我们想要运行[目标文件](slurm.py)，在一张GPU上我们想要同时进行两个目标文件的运行，通过参数index和cuda来确定自己使用的显卡和编号。

为了让[目标文件](slurm.py)能够得到自己的编号和显卡，同时也让该节点可以同时运行多个进程，我们使用[脚本](run.sh)来进行控制，在[范例脚本](slurm.slurm)中，通过后台运行[脚本](run.sh)的方式来进行多进程操作，用wait命令来让slurm任务等待后台进程结束后退出。

因此在这个例子中，通过sbatch slurm.slurm命令：

1. 会向集群提交1个节点，4张GPU，64个CPU，256G内存的资源申请。
2. 申请到资源后，激活conda环境，并依次在后台挂起run.sh脚本指令，并在wait命令处等待后台进程结束
3. run.sh的命令在后台执行main.py脚本，并提供对应的index和cuda参数
4. main.py脚本全部执行完毕后，wait命令确定后台进程执行完，结束任务，释放资源

### 使用array来提交多个任务

如果因为一些原因需要申请大批量小资源的任务（如处理数据），最简单的方法就是使用bash脚本进行大量申请，但是这样会导致产出大量的Pending任务，可能会导致队列饱和，同一组下其他用户无法提交任务。

因此我们对于大量的相似任务，可以选择使用job array来减少等待队列任务而达到同样的效果。

使用job array，相当于申请了一系列的任务，但是只是相当于提交了一次申请，每次符合其中一个job array的时候就申请一次，直到申请完整个array为止，对于每一个单独的job array task，相当于单次申请。

```
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
#SBATCH --array=0-15                # 申请了一共16个job array task



export NCCL_DEBUG=INFO                      # Optional: for debugging NCCL issues
export MASTER_ADDR=$(scontrol show hostname | head -n 1)  # Get the master node's hostname
export MASTER_PORT=$((29501))                    # Port for distributed communication

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
```


## 一些tips

如果你在跑的时候发现几乎所有流程都对，显示也对，但是python就是不输出内容，导致从观察结果来看似乎任务在空跑。

那么可能是python的输出在缓冲区并未输出。

可以尝试

```
print('something')
print('other things')
print('final thing', flush=True)

```

在最后一个输出加上flush选项，强制清空缓存区输出，而不是等到缓存区满后再输出。