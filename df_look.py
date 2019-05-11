import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('test.csv')
df = df[df.result >= 64]

print df.count()

df.hist()
plt.show()

print df