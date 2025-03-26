import torch
import os
import matplotlib.pyplot as plt

def vis_acc(accuracy1,accuracy2):
    # 示例数据（替换成你的实际数据）
    accuracy1 = torch.randn(1000, dtype=torch.float32)
    accuracy2 = torch.randn(1000, dtype=torch.float32)

    min_val = 0.0
    max_val = 1.0
    num_bins = 20

    # 修正1：生成正确的分界点，覆盖0.0到1.0的20个等宽区间
    bins = torch.linspace(min_val, max_val, num_bins + 1)  # 21个分界点，形成20个左闭右开区间
    bin_width = (max_val - min_val) / num_bins  # 每个区间的宽度

    # 修正2：计算分桶索引，不需要减1，并限制索引范围为0~num_bins
    indices1 = torch.bucketize(accuracy1, bins)
    indices1 = torch.clamp(indices1, 0, num_bins)  # 避免浮点误差导致的越界
    indices2 = torch.bucketize(accuracy2, bins)
    indices2 = torch.clamp(indices2, 0, num_bins)

    # 统计每个区间的数量，minlength设置为num_bins +1以包含所有可能索引
    counts1 = torch.bincount(indices1, minlength=num_bins + 1).tolist()
    counts2 = torch.bincount(indices2, minlength=num_bins + 1).tolist()

    # 可视化对比（修正3：只取前num_bins个有效区间）
    plt.figure(figsize=(18, 6))
    x_positions = bins[:-1]  # 区间左端点作为标签位置
    bar_width = bin_width / 5  # 调整柱子宽度以适配标签

    bar1 = plt.bar(
        x_positions - bar_width/2, counts1[:num_bins],
        width=bar_width, label='Tensor 1', color='#2c3e50'
    )
    bar2 = plt.bar(
        x_positions + bar_width/2, counts2[:num_bins],
        width=bar_width, label='Tensor 2', color='#3498db'
    )

    # 添加数值标签
    for i in range(num_bins):
        plt.text(
            x_positions[i], bar1[i].get_height() + 0.5,
            f"{counts1[i]:d}", ha='center', va='bottom'
        )
        plt.text(
            x_positions[i], bar2[i].get_height() + 0.5,
            f"{counts2[i]:d}", ha='center', va='bottom'
        )

    # 设置坐标轴和标题
    plt.title("双Tensor准确率区间对比", fontsize=14)
    plt.xlabel("准确率区间", fontsize=12)
    plt.ylabel("类别数量", fontsize=12)
    # plt.xticks(rotation=45, labels=[f"[{b:.2f}, {b + bin_width:.2f})" for b in x_positions])
    plt.legend()
    plt.grid(axis='y', alpha=0.7)
    # plt.show()
    file_path = os.path.join('.', 'test.png')
    plt.savefig(
        file_path,
        dpi=300,          # 分辨率（默认100，300为高清）
        bbox_inches='tight',  # 去除多余空白
        facecolor='white'     # 背景颜色
    )

def main():
    vis_acc(acc1,acc2)

if __name__ == '__main__':
    main()