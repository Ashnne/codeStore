import cv2
import imageio.v2 as imageio

# imgs 是一个列表，列表中每个元素是一个图片
# img [h,w,3] (0-255) np.uint8
def trans_GIF(imgs, output_gif, width=None, height=None):
    # 配置参数
    fps = 10             # 每秒的帧数

    # 读取所有图片并统一尺寸
    gif_images = []
    for img in imgs:
        if width is not None and height is not None:
            img=cv2.resize(img,(512,512))
        gif_images.append(img)

    # 3. 保存为GIF
    print(f"找到 {len(gif_images)} 张图片,开始生成GIF...")
    imageio.mimsave(output_gif,gif_images,fps=fps,loop=0)


# img [h,w,3] (0-255) np.uint8
# points [n,2] (x,y)
def draw_line(img, points, color=(0,0,255), size=4):
    for i in range(len(points)-1):
        img = cv2.line(img,points[i],points[i+1],tuple([int(x) for x in color]),size)
    return img

