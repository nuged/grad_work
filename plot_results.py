import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('tmp.txt', header=None)
test = [0.4538624227046967, 0.3556966006755829, 0.3060252904891968]
df = df.drop(360, axis=1)
data = df.values

data = data.reshape(-1)
window_width = 8
cumsum_vec = np.cumsum(np.insert(data, 0, 0))
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width

xticks = np.arange(0, 27000, 25)
plt.figure(figsize=(7, 7))
plt.plot(xticks[window_width - 1:], ma_vec, zorder=1, label='Training')
plt.scatter([8999, 17999, 26999], test, color='orange', s=100, marker='h', zorder=2, label='Testing')
plt.xlabel('iterations', fontsize=18)
plt.ylabel('Cross entropy loss', fontsize=18)
plt.title('Manual pattern', fontsize=24)
plt.legend(fontsize=18)
plt.grid()
plt.show()
