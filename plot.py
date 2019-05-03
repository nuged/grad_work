import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('TM_test_0.csv')
df = df[df.result >= 38.]

df.hist()
plt.title('total')
plt.show()

df = df[df.result > 38.1]

df.hist()
plt.title('two best')
plt.show()

gr = df.groupby('result')

for g in gr:
    value = g[0]
    data = g[1]
    data.hist()
    plt.title(str(value))
    plt.show()