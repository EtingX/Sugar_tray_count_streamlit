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
        return []

    # 只问一次总的 seedlings 数
    seedlings_counts = input("How many seedlings in all boxes? ")

    # 返回一个元组
    results = [(boxes, seedlings_counts)]
    return results

# 使用：
results = draw_boxes_and_seedlings(r"D:\SRA dainel\images\4.png")
print("最终返回的列表：", results)


