import numpy as np
# import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import cv2
import matplotlib.pyplot as plt
import os


# inputs N, D \
# labels N, 1
# dimension 2 for visulization on 2D and 3 for 3D
def t_SNE_visulization(inputs, labels, dimension=2):
    t_sne_features = TSNE(n_components=dimension, learning_rate='auto', init='pca').fit_transform(inputs)
    # plt.scatter(x=t_sne_features[:, 0], y=t_sne_features[:, 1], c=labels, cmap='jet')
    result = np.concat((t_sne_features,labels.reshape(*labels.shape,1)),axis=1)
    return result


def visual_and_save(inputs, labels, save_dir, filename, title="visual",dimension=2):
    
    data = t_SNE_visulization(inputs,labels,dimension)

    # 创建散点图
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        data[:, 0], data[:, 1],
        c=data[:, 2], cmap='tab10', s=20, alpha=0.7,
        edgecolors='w', linewidths=0.5
    )

    # 添加颜色条和标签
    cbar = plt.colorbar(scatter, ticks=np.unique(data[:, 2]))
    cbar.set_label('Class Label')

    # 设置标题和坐标轴
    plt.title(title)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # 定义保存路径
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）

    # 保存图像（支持PNG、JPG、PDF等格式）
    file_path = os.path.join(save_dir, filename)
    plt.savefig(
        file_path,
        dpi=300,          # 分辨率（默认100，300为高清）
        bbox_inches='tight',  # 去除多余空白
        facecolor='white'     # 背景颜色
    )

    # 显示图像（可选）
    # plt.show()

def main():
    path = '/Users/wujiefeng/Downloads/test'
    result = torch.load(path,map_location='cpu')
    feature = np.array(result['feature'])
    labels = np.array(result['classifier'])
    linear = np.array(result['linear'])

    visual_and_save(feature,labels,'save_dir','feature','feature')
    visual_and_save(linear,labels,'save_dir','linear','linear')




if __name__ == "__main__":
    main()