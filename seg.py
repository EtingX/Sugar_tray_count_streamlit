from plantcv import plantcv as pcv
import numpy as np


# 读图（返回RGB）
img, path, filename = pcv.readimage(r"D:\SRA dainel\images\4.png")

# --- 提取 H 和 a 通道 ---
h = pcv.rgb2gray_hsv(rgb_img=img, channel='h')
a = pcv.rgb2gray_lab(rgb_img=img, channel='a')

# --- 分别阈值化 ---
# H通道：去掉高于70的 (保留 <=70)
mask_h = pcv.threshold.binary(gray_img=h, threshold=70, object_type='dark')

# a通道：去掉高于120的 (保留 <=120)
mask_a = pcv.threshold.binary(gray_img=a, threshold=120, object_type='dark')

# --- 合并两个mask：取交集 (即两条件都满足) ---
mask_combined = pcv.logical_and(mask_h, mask_a)

# # --- 可视化 ---
# pcv.plot_image(h)                # H通道
# pcv.plot_image(a)                # a通道
# pcv.plot_image(mask_h)           # H阈值mask
# pcv.plot_image(mask_a)           # a阈值mask
# pcv.plot_image(mask_combined)    # 最终mask

clean = pcv.fill(bin_img=mask_combined, size=1000)

# pcv.plot_image(clean)

masked_img = pcv.apply_mask(img=img, mask=clean, mask_color='black')

# 显示结果
pcv.plot_image(masked_img)



white_pixels = np.sum(clean == 255)
print("White pixels:", white_pixels)



