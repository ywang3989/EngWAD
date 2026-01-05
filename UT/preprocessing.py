import numpy as np
import matplotlib.pyplot as plt


number_array = 64
number_array_batch = 4

date = '1122'
depth = 251  # 320 for 1128, 344 for 0805 & 0806, 267 for 0813 & 0821 & 0823, 251 for 1122
sample_num = 64
fw_l = 20
fw_u = 60
N = 1000
T = 1e-8
whether_plot_comparison = True

ascan_train = np.genfromtxt('ascan_train_' + date + '.csv', delimiter=',').astype('f4')
ascan_panel = np.genfromtxt('ascan_panel_' + date + '.csv', delimiter=',').astype('f4')

ascan_fft_train = np.zeros((ascan_train.shape[0], 5))
for i, ascan in enumerate(ascan_train):
    fft_signal = np.abs(np.fft.fft(ascan[fw_l:fw_u], n=N))
    # freq = np.fft.fftfreq(N, d=T)
    # plt.plot(freq[0:N//2], fft_signal[0:N//2])
    # plt.show()
    ascan_fft_train[i] = fft_signal[48:53]

ascan_fft_panel = np.zeros((ascan_panel.shape[0], 5))
for i, ascan in enumerate(ascan_panel):
    fft_signal = np.abs(np.fft.fft(ascan[fw_l:fw_u], n=N))
    ascan_fft_panel[i] = fft_signal[48:53]

augment_index = np.zeros(ascan_train.shape[0]).astype(np.int8)
ascan_train_aug = np.zeros((ascan_train.shape[0], depth+1))
reflection_rates = np.zeros(ascan_train.shape[0])
for i, fft_train in enumerate(ascan_fft_train):
    dist = np.zeros(ascan_fft_panel.shape[0])
    for j, fft_panel in enumerate(ascan_fft_panel):
        dist[j] = np.linalg.norm(fft_train - fft_panel)
    augment_index[i] = np.argmin(dist)  # Searching for closest one
    
    # Refelction rate R
    pos_train = np.argmax(ascan_train[i, 150:200])
    int_l_train = pos_train + 120  # +150-30
    int_u_train = int(np.min([pos_train+180, 250]))
    fft_train_bw = np.abs(np.fft.fft(ascan_train[i, int_l_train:int_u_train], n=N))[50]

    pos_panel = np.argmax(ascan_panel[augment_index[i], 150:200])
    int_l_panel = pos_panel + 120  # +150-30
    int_u_panel = int(np.min([pos_panel+180, 250]))
    fft_panel_bw = np.abs(np.fft.fft(ascan_panel[augment_index[i], int_l_panel:int_u_panel], n=N))[50]

    reflection_rates[i] = fft_train_bw / fft_panel_bw
    
    ascan_train_aug[i, 0:depth] = ascan_panel[augment_index[i]]

    if whether_plot_comparison:
        plt.plot(ascan_train[i], label='train')
        plt.plot(ascan_panel[augment_index[i]], label='augment')
        plt.title(np.array2string(np.round(reflection_rates[i], 3)))
        plt.legend()
        plt.show()

# Normalization
# reflection_rates = 2 * (reflection_rates-np.min(reflection_rates)) / (np.max(reflection_rates)-np.min(reflection_rates)) - 1
ascan_train_aug[:, depth] = reflection_rates

np.savetxt('ascan_train_aug_' + date + '.csv', ascan_train_aug, delimiter = ',')

