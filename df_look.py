import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv('TM_results.csv')

df = df[df.result > 28]
df.drop(['time', 'result'], axis=1, inplace=True)

df2 = df.copy()
df3 = df.copy()
df2.cellsPerColumn = 4
df3.columnCount = 4000

df = df.append(df2).append(df3)
print df

df.to_csv('test_params.csv', ignore_index=True)