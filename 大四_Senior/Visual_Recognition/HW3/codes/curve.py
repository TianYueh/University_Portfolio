import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔
df = pd.read_csv('training_log.csv')

# 畫出 Train Loss 和 Val Loss 折線圖
plt.figure(figsize=(10, 6))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss')
plt.plot(df['Epoch'], df['Val Loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
