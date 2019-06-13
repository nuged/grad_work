# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_accuracy(data, mode='train'):
    xs = np.arange(1, 6)
    plt.figure(figsize=(10, 10))
    for key in data:
        if key == 'MC':
            c = 'blue'
        elif key == '31223':
            c = 'red'
        else:
            c = None
        if key in ['MC', '31223']:
            plt.plot(xs, data[key][mode + '_accs'], label=key, linewidth=3, c=c)
        else:
            plt.plot(xs, data[key][mode + '_accs'], label=key, linewidth=1, alpha=0.75, c=c)
    plt.xticks(xs)
    if mode == 'train':
        ys = np.arange(75, 96, 2)
    else:
        ys = np.arange(83, 96, 1)

    plt.yticks(ys)
    plt.title(mode, fontsize=24)
    plt.xlabel('epoch', fontsize=18)
    plt.ylabel('accuracy score', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid()
    plt.show()

data = {}
for folder in os.listdir('scores'):
    path = os.path.join('scores', folder)
    folder_data = {}
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as f:
            file_data = f.read()
            if file_data.startswith('['):
                file_data = file_data[1:-2]
            file_data = file_data.split(',')
            if '\n' in file_data:
                file_data.remove('\n')
            file_data = map(float, file_data)
            folder_data[file[:-4]] = file_data
    n_iters = len(folder_data['train_losses']) // 5

    data[folder] = folder_data

    N = 10
    cumsum = np.cumsum(np.insert(folder_data['train_losses'], 0, 0))
    train_losses = (cumsum[N:] - cumsum[:-N]) / float(N)
    train_losses = np.clip(train_losses, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.plot(np.arange(N, n_iters * 5 + 1), train_losses, zorder=0,label='train')
    xticks = np.arange(0, n_iters * 5 + 2, n_iters)
    plt.xticks(xticks)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.plot([n_iters * i for i in range(1, 6)], folder_data['test_losses'], c='red', marker='*', zorder=1,
             markersize=15, label='test')
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('cross entropy loss', fontsize=18)
    plt.legend(fontsize=18)
    plt.title(folder, fontsize=24)
    plt.show()

    del folder_data['train_losses']

plot_accuracy(data)
plot_accuracy(data, mode='test')
