# 可视化专区

在debug还是在展示论文的时候我们都需要各种各样的可视化方法，我把我用到的可视化都转成即插即用的可视化函数

[t-SNE](t-SNE) :t-SNE函数和PCA一样可以将高维feature降到二维平面或者三维空间来进行可视化，适用查看feature的分布。

[vis3d](vis3d.py) :对空间上的点进行可视化，还可以将feature值转成颜色。

[viskeyword](vis_keyword.py) :对有权重的句子先进行拆解成关键词，再用词云进行可视化

[gif](gif.py) : 将图片转化为gif输出，还有在图片上画线