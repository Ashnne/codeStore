import torchvision
import cv2
# Image,imageio 读取的图片
# [h,w,c]
# img[0  ,  0] 表示在最左上角
#   纵坐标 横坐标
# 蓝
# img_color[100:200,100:200] = [0,0,255]
# 绿
# img_color[200:300,100:200] = [0,255,0]
# 红
# img_color[300:400,100:200] = [255,0,0]
# 黄
# img_color[400:500,100:200] = [255,255,0]
# 粉
# img_color[500:600,100:200] = [255,0,255]
# 青
# img_color[600:700,100:200] = [0,255,255]



# 保存numpy array的图像

# 保存torch tensor的图像

# 读取图像

# 图像裁剪
# loc_h：原点离左上角的纵向处理
# loc_w：原点离左上角的横向距离
# h：从原点向下裁剪的距离
# w：从原点向右裁剪的距离
# (0,0)
    
#         (loc_h,loc_w)       (loc_h,loc_w+w)

#                     crop zone

#         (loc_h+h,loc_w)     (loc_h+h,loc_w+w)



def img_crop(img,loc_h,loc_w,h,w):
    return torchvision.transforms.functional.crop(img,loc_h,loc_w,h,w)

# 图像resize

def resize(img, h, w):
    return cv2.resize(img, (w,h))

# gif使用

