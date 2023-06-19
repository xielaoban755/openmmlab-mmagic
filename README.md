# openmmlab-mmagic
1.文生图或网上下载一张毛胚房图片
![R](https://github.com/xielaoban755/openmmlab-mmagic/assets/114243452/e08d58be-18d3-4542-9863-e6ff8d2a02f7)

2.使用opencv将图像转为边缘检测图
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image
from mmagic.registry import MODELS
from mmagic.utils import register_all_modules
register_all_modules()
import cv2

# 读取输入图像
img = cv2.imread(r"C:\Users\17219\Desktop\R.jpg", cv2.IMREAD_GRAYSCALE)

# 应用Canny边缘检测算法
edges = cv2.Canny(img, 100, 200)

# 保存输出图像
cv2.imwrite('output_image.jpg', edges)

![output_image](https://github.com/xielaoban755/openmmlab-mmagic/assets/114243452/3cc37397-e11b-43f4-8e36-0de424c5b132)


3.加载模型
cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')
controlnet = MODELS.build(cfg.model).cuda()

4.设置咒语生成图像
control_img = mmcv.imread(r"F:\openmmlab\mmagic\output_image.jpg")
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)
type(control)
prompt = 'Room with pastoral style.'
output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'outputs/sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'outputs/control_{idx}.png')

因本人计算机显存不够，加载模型失败无法生成图像
