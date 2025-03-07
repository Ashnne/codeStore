
# 单机多 GPU（例如 4 个 GPU）
# 示例运行命令
torchrun --nproc_per_node=4 train.py \
  --pretrained_ckpt ./pretrained/model.pt \
  --save_dir my_checkpoints \
  --epochs 50 \
  --save_interval 5

# 多机多 GPU（需要指定主节点地址）
# 节点0（主节点）
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=192.168.1.1 --master_port=1234 train.py

# 节点1
# torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=192.168.1.1 --master_port=1234 train.py