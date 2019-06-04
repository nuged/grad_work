import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


mctrain = [48.1, 72.7, 78.3] #4096
mctest = [74.3, 81.3, 81.0]

matrain = [43.4, 75.0, 80] # 2048
matest = [70, 81, 82.3]


plt.plot([1, 2, 3], mctrain, label='MCS train', linewidth=3)
plt.plot([1, 2, 3], mctest, label='MCS test', linewidth=3)
plt.plot([1, 2, 3], matrain, label='Manual pattern train', linewidth=3)
plt.plot([1, 2, 3], matest, label='Manual pattern test', linewidth=3)
plt.yticks(np.arange(60, 100, 5))
plt.xticks([1, 2, 3])
plt.xlabel('epoch', fontsize=18)
plt.ylabel('Accuracy', fontsize=18)
plt.title('Monte Carlo vs Manual pattern', fontsize=24)
plt.grid()
plt.legend(fontsize=18)
plt.show()
