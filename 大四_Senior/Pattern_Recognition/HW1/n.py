import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# 參數設定
years = np.arange(2012, 2026)
width = len(years) - 1
height = 300  # 垂直解析度

# 顏色定義
green = np.array(to_rgb("#00aa00"))
white = np.array(to_rgb("#ffffff"))
blue = np.array(to_rgb("#0000aa"))
tiffany = np.array(to_rgb("#0abab5"))

# 建立圖片陣列 (height, width, 3)
image = np.zeros((height, width, 3))

# 定義漸層範圍
top = int(height * 0.72)
mid = int(height * 0.34)

area = 30

for x in range(width):
    for y in range(height):
        if y >= top:
            color = green
        elif y >= top - area:
            # 綠 -> 中間色
            ratio = (top-y) / area
            mid_color = white if years[x] < 2019 else tiffany
            color = (1-ratio) * green + (ratio) * mid_color
        elif y >= mid:
            color = white if years[x] < 2019 else tiffany
        elif y >= mid - area:
            # 中間色 -> 藍
            ratio = (mid-y) / area
            mid_color = white if years[x] < 2019 else tiffany
            color = (1-ratio) * mid_color + (ratio) * blue
        else:
            color = blue
        image[y, x] = color

# 繪製圖像
fig, ax = plt.subplots(figsize=(12, 2))
ax.imshow(image, aspect='auto', extent=[2012, 2025, 0, 1])
ax.set_xticks(years)
ax.set_yticks([])
ax.set_title("Taiwanese Political Status (2012-2025)", fontsize=16)
plt.tight_layout()
plt.show()
