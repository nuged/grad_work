import numpy as np
import matplotlib.pyplot as plt

def plot_img(image):
    image = image.reshape(10, 10)
    plt.imshow(image, cmap='gray')
    plt.xticks(np.arange(-0.5, 10, 1), np.arange(11))
    plt.yticks(np.arange(-.5, 10, 1), np.arange(11))
    plt.grid(linewidth=2, c='blue')
    plt.show()


def plot_sp(sp_out):
    plt.figure(figsize=(16, 9))
    sp_out = sp_out.reshape(64, 64)
    plt.imshow(sp_out, cmap='gray')
    plt.xticks(np.arange(-0.5, 64, 1), np.arange(65))
    plt.yticks(np.arange(-.5, 32, 1), np.arange(33))
    plt.grid(linewidth=2, c='blue')
    plt.show()


def plot_tm(tm_out):
    plt.figure(figsize=(70, 45))
    tm_out = tm_out.reshape(64, 256)
    plt.imshow(tm_out, cmap='gray', aspect="auto")
    plt.xticks(np.arange(-0.5, 256, 4))
    plt.yticks(np.arange(-.5, 64, 1))
    plt.grid(linewidth=8, c='green')
    plt.show()

if __name__ == '__main__':
    with open('logs/sp_in.log', 'r') as sens, open('logs/sp_out.log', 'r') as sp,\
            open('logs/tm_out.log', 'r') as tm:
        for i in range(100):
            sens_data = sens.readline()
            sp_data = sp.readline()
            tm_data = tm.readline()

            sens_idx = map(int, sens_data.split()[1:])
            sp_idx = map(int, sp_data.split()[1:])
            tm_idx = map(int, tm_data.split()[1:])

            sens_data = np.zeros(int(sens_data.split()[0]))
            sp_data = np.zeros(int(sp_data.split()[0]))
            tm_data = np.zeros(int(tm_data.split()[0]))

            sens_data[sens_idx] = 1
            sp_data[sp_idx] = 1
            tm_data[tm_idx] = 1

            plot_img(sens_data)
            plot_sp(sp_data)
            plot_tm(tm_data)

            raw_input("Press Enter to continue...")