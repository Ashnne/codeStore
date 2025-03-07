import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os

from dataset import prepare_dataloader

def setup(rank, world_size):
    # 初始化进程组，使用 NCCL（推荐 GPU）或 GLOO（CPU）
    dist.init_process_group(
        backend="nccl",  # 或 "gloo"（CPU）
        init_method="env://",  # 从环境变量读取 MASTER_ADDR 和 MASTER_PORT
        rank=rank,
        world_size=world_size
    )
    # 设置当前进程的 CUDA 设备（仅限 GPU 训练）
    torch.cuda.set_device(rank)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

def save_checkpoint(model, optimizer, epoch, save_dir):
    """只在主进程保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # 保存最新检查点
    latest_path = os.path.join(save_dir, "latest.pt")
    # 按epoch保存
    epoch_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(checkpoint, latest_path)
    torch.save(checkpoint, epoch_path)
    print(f"Checkpoint saved at epoch {epoch}")

# 6. 训练函数
def train(rank, world_size, args):
    # 初始化分布式环境
    setup(rank, world_size)
    
    # 创建模型并移至GPU
    model = MyModel().to(rank)
    
    # 加载预训练权重（只在主进程加载）
    if args.pretrained_ckpt:
        if rank == 0:
            try:
                state_dict = torch.load(
                    args.pretrained_ckpt, 
                    map_location=f"cuda:{rank}"
                )
                # 处理可能的module前缀（如果预训练模型是DDP保存的）
                state_dict = {k.replace("module.", "", 1): v 
                             for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                print(f"Loaded pretrained checkpoint from {args.pretrained_ckpt}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
        # 等待所有进程同步
        dist.barrier()
    
    # 包装为DDP模型
    model = DDP(model, device_ids=[rank])
    
    # 准备数据
    dataloader = prepare_dataloader(rank, world_size, args.batch_size)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(args.epochs):
        dataloader.sampler.set_epoch(epoch)
        
        for x, y in dataloader:
            x, y = x.to(rank), y.to(rank)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # 每个epoch结束后保存检查点（只在主进程）
        if rank == 0 and (epoch % args.save_interval == 0 or epoch == args.epochs - 1):
            save_checkpoint(model, optimizer, epoch, args.save_dir)
        
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
    
    # 清理进程组
    dist.destroy_process_group()

# 7. 主函数
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_interval", type=int, default=1)
    args = parser.parse_args()
    
    # 从环境变量获取分布式参数
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    train(rank, world_size, args)

if __name__ == "__main__":
    main()