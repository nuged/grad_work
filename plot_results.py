import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

size = 2

df = pd.read_csv('logs/historic/size_%d.txt' % size, delim_whitespace=True, header=None)
data = df[8].values
val_data = df[df[9] == 'eval'][8].values
grid = np.arange(0, 60, 5)

plt.figure(figsize=(8, 5))
plt.plot(data, label='training')
plt.scatter(grid, val_data, c='orange', marker='*', s=200, label='evaluation')
plt.xticks(np.arange(0, 65, 5))
plt.yticks(np.arange(10, 65, 5))
plt.xlabel('Num episodes')
plt.ylabel('Accuracy score')
plt.title('History size: %d' % size)
plt.legend()
plt.grid()
plt.show()