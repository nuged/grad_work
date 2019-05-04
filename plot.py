import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('SP1_test_0.csv')
df = df[df.result >= 45]
df['ratio'] = df.nActCol / df.colCnt

print df

df.hist()
plt.show()