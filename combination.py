import os

from plantcv import plantcv as pcv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def draw_boxes_and_seedlings(image_path):
    """
    交互式画多个矩形框，关闭窗口后只问一次全部seedlings总数。
    返回 [(coords列表, 总seedlings数)]
    """
    img = plt.imread(image_path)

    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title("Drag the mouse to draw rectangles (multiple allowed), press 'q' to finish.")
    boxes = []  # 存储框坐标

    # 画框回调
    def onselect(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return
        x1, x2 = sorted([int(round(x1)), int(round(x2))])
        y1, y2 = sorted([int(round(y1)), int(round(y2))])
        boxes.append((x1, y1, x2, y2))
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                             fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        fig.canvas.draw_idle()

    # 按 q 关闭窗口
    def on_key(event):
        if event.key == "q":
            plt.close(fig)

    selector = RectangleSelector(ax, onselect,
                                 useblit=False,
                                 button=[1],
                                 minspanx=2, minspany=2,
                                 spancoords='pixels',
                                 interactive=False)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()

    if not boxes:
        print("No boxes drawn.")
        return None

    # 只问一次总的 seedlings 数
    seedlings_counts = input("How many seedlings in all boxes? ")

    # 返回一个元组
    results = [(boxes, seedlings_counts)]
    return results

def box_white_pixels(bin_img, box):
    """bin_img: 0/255 的二值图; box: (x1,y1,x2,y2)"""
    h, w = bin_img.shape[:2]
    x1, y1, x2, y2 = box
    # 边界安全裁剪（避免越界）
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return 0
    roi = bin_img[y1:y2, x1:x2]
    # 白像素计数（非零）
    return int(np.count_nonzero(roi))



def segmentation(image_path):
    # 读图（返回RGB）
    img, path, filename = pcv.readimage(image_path)

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

    # masked_img = pcv.apply_mask(img=img, mask=clean, mask_color='black')

    # # 显示结果
    # pcv.plot_image(masked_img)

    white_pixels = np.sum(clean == 255)
    return white_pixels, clean
    # print("White pixels:", white_pixels)


tmp_avg_white_per_seedling = 0

image_folder = r'D:\SRA dainel\images'
total_seedlings_list = []
sample_id_list = []

for img in os.listdir(image_folder):
    img_path = os.path.join(image_folder,img)
    sample_id = input("What is your sample ID? (Press Enter to use image name) ").strip()

    # If no input, use image file name without extension
    if not sample_id:
        sample_id = os.path.splitext(os.path.basename(img_path))[0]

    sample_id_list.append(sample_id)

    if tmp_avg_white_per_seedling == 0:
        # Ensure tmp_avg_white_per_seedling is valid (>0) before proceeding
        while tmp_avg_white_per_seedling == 0:
            # 1) Ask user to draw boxes and input total seedlings count
            record = draw_boxes_and_seedlings(img_path)

            # ---- Validate record before using it ----
            # record structure should be: [(boxes, seedlings_str)]
            if not record or not record[0] or not record[0][0]:
                print("No boxes drawn. Please draw at least one box.")
                continue

            boxes, seedlings_str = record[0]

            # Clean and validate seedlings count
            seedlings_str = str(seedlings_str).strip()
            if not seedlings_str.isdigit():
                print(f"Invalid seedlings count '{seedlings_str}'. Please enter a positive integer.")
                continue

            seedlings_count = int(seedlings_str)
            if seedlings_count <= 0:
                print("Seedlings count must be > 0. Please draw and input again.")
                continue

            # 2) Perform segmentation to obtain binary mask and total white pixels
            white_pixels, clean = segmentation(img_path)

            # 3) Calculate white pixels inside each drawn box
            per_box_whites = [box_white_pixels(clean, b) for b in boxes]
            total_white = sum(per_box_whites)

            # 4) Compute average white pixels per seedling if possible
            if total_white > 0:
                avg_white_per_seedling = total_white / seedlings_count
                tmp_avg_white_per_seedling = avg_white_per_seedling
            else:
                print("No white pixels found inside boxes. Please re-label boxes.")
                # keep tmp_avg_white_per_seedling == 0 to repeat the loop
                continue

        # 5) Once tmp_avg_white_per_seedling is valid, compute seedlings number
        each_seedlings_num = int(round(white_pixels / tmp_avg_white_per_seedling))
        total_seedlings_list.append(each_seedlings_num)


    else:
        record = draw_boxes_and_seedlings(img_path)
        if record == None:
            white_pixels, clean = segmentation(img_path)
            each_seedlings_num = int(round(white_pixels / tmp_avg_white_per_seedling, 0))
            total_seedlings_list.append(each_seedlings_num)

        else:
            white_pixels, clean = segmentation(img_path)

            # 拆出 boxes 和 seedlings_count
            boxes, seedlings_str = record[0]
            seedlings_count = int(seedlings_str)

            per_box_whites = [box_white_pixels(clean, b) for b in boxes]
            total_white = sum(per_box_whites)

            avg_white_per_seedling = total_white / seedlings_count
            tmp_avg_white_per_seedling = avg_white_per_seedling

            each_seedlings_num = int(round(white_pixels / tmp_avg_white_per_seedling, 0))
            total_seedlings_list.append(each_seedlings_num)

print(total_seedlings_list)
print(sample_id_list)


