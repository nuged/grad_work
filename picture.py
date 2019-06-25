import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, numpy as np
from PIL import Image

paths = {
'0':	[5, 9, 8, 4, 5, 9] ,
'1':	[5, 1, 2, 1, 2, 1] ,
'2':	[5, 6, 10, 9, 5, 6] ,
'3':	[5, 6, 10, 9, 8, 9] ,
'4':	[5, 9, 10, 6, 5, 9] ,
'5':	[5, 9, 10, 6, 5, 9] ,
'6':	[5, 6, 10, 6, 10, 6] ,
'7':	[5, 6, 10, 9, 10, 9] ,
'8':	[5, 4, 8, 9, 8, 9] ,
'9':	[5, 6, 10, 9, 5, 6] ,
}

directory = 'mnist/small_training'
fig, axes = plt.subplots(4, 3, figsize=(10, 10), sharex='all', sharey='all')

for dir in os.listdir(directory):
    avg = np.zeros((28, 28))
    counter = 0
    for fimg in os.listdir(os.path.join(directory, dir)):
       img = Image.open(os.path.join(directory, dir, fimg))
       pix = np.array(img) / 255.
       pix = pix.round()
       avg += pix
       counter += 1
    avg /= counter
    cat = int(dir)
    axes[cat // 3, cat % 3].imshow(avg, cmap='Greys')

    texts = []
    for i, state in enumerate(paths[dir]):
        position = state % 4 * 7, state // 4 * 7
        #position = map(lambda x : 28 - x, position)
        rect = patches.Rectangle(position, 10, 10, linewidth=1, edgecolor='r', facecolor='none')
        axes[cat // 3, cat % 3].add_patch(rect)
        axes[cat // 3, cat % 3].annotate(i, xy=map(lambda x : x + 5, position), xycoords='data',
                    xytext=[position[0] + 5, position[1] + 5 + i % 3], color='red', fontsize=12
                )

for i in range(4):
    for j in range(3):
        axes[i, j].set_xticklabels([])
        axes[i, j].set_yticklabels([])
        axes[i, j].set_aspect('equal')
fig.subplots_adjust(hspace=0, wspace=0.01)
plt.show()