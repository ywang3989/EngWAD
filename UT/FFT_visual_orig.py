import numpy as np
import os
import os.path
import matplotlib.pyplot as plt

def read_file(file_path, number_array, number_array_batch):
    data = np.genfromtxt(file_path, delimiter=',')
    data = data[1:, 2:]
    beam_num = number_array - number_array_batch + 1
    frame_num = int(data.size / (beam_num * data.shape[1]))
    
    cscan_data = np.zeros((beam_num, frame_num, data.shape[1]))
    for j in range(frame_num):
        cscan_data[:, j, :] = data[j*beam_num:(j+1)*beam_num, :]

    return cscan_data


data_dir = os.getcwd()
data_dir = data_dir.replace('Codes', 'Data')

number_array = 64
number_array_batch = 4

date = '1122'
depth = 251  # 320 for 1128, 344 for 0805 & 0806, 267 for 0813 & 0821 & 0823, 251 for 1122
sample_num = 64
intersec_l = 160
intersec_u = 230
L = 1000

nbeam_U = []
nbeam_L = []
nscan_U = []
nscan_L = []
A = []
for k in range(sample_num):
    if k < 10:
        data_filename = '00' + str(k) + '.csv'
    else:
        data_filename = '0' + str(k) + '.csv'
    
    if date == '1128':
        data_file_path = os.path.join(data_dir + '\\2023' + date, data_filename)
    else:
        data_file_path = os.path.join(data_dir + '\\2024' + date, data_filename)
    cscan_data = read_file(data_file_path, number_array, number_array_batch)
    cscan_data = (cscan_data - 127) / 127  # Normalization

    # Cropping
    nbeam_U.append(cscan_data.shape[0])
    nbeam_L.append(0)
    nscan_U.append(cscan_data.shape[1])
    nscan_L.append(0)

    # FFT
    for i in range(cscan_data.shape[0]):
        for j in range(cscan_data.shape[1]):
            pos = np.argmax(cscan_data[i, j, 150:200])
            intersec_l = pos + 140  # +150-10
            intersec_u = int(np.min([pos + 200, 250]))  # +150+50
            ascan = cscan_data[i, j, intersec_l:intersec_u]
            fft_signal = np.abs(np.fft.fft(ascan, n=L))
            A.append(np.max(fft_signal[0:int(L/2)]))
    
A = np.array(A)

# Plot
k = 0
count = 0
start_temp = 0
ratio = 1/1.5
value_min = np.min(A)
value_max = np.max(A)
plt.style.use('classic')
if date == '1128':  # Without mouse encoders
    fig, axes = plt.subplots(nrows=5, ncols=3)
    for ax in axes.flat:
        nbeam = nbeam_U[k] - nbeam_L[k]
        nscan = nscan_U[k] - nscan_L[k]
        score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
        im = ax.imshow(score, vmin=value_min, vmax=value_max)

        ax.set_title('#' + str(k))
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        k += 1
        start_temp += nbeam * nscan
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '1122':
    fig, axes = plt.subplots(nrows=9, ncols=8)
    for ax in axes.flat:
        if k < sample_num and count not in [14, 15, 22, 23, 55, 61, 62, 63]:
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            ax.set_title('#' + str(k))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            k += 1
            start_temp += nbeam * nscan
        
        count += 1
    axes[1, 6].axis('off')
    axes[1, 7].axis('off')
    axes[2, 6].axis('off')
    axes[2, 7].axis('off')
    axes[6, 7].axis('off')
    axes[7, 5].axis('off')
    axes[7, 6].axis('off')
    axes[7, 7].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0813':
    fig, axes = plt.subplots(nrows=4, ncols=10)
    for ax in axes.flat:
        if k < sample_num and count != 28 and count != 29 and count != 38 and count != 39:
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            ax.set_title('#' + str(k))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            ax.axis('off')
            
            k += 1
            start_temp += nbeam * nscan
        
        count += 1
    axes[2, 8].axis('off')
    axes[2, 9].axis('off')
    axes[3, 8].axis('off')
    axes[3, 9].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0821' or date == '0823':
    fig, axes = plt.subplots(nrows=6, ncols=8)
    for ax in axes.flat:
        if k < sample_num and count not in [15, 22, 23, 30, 31, 37, 38, 39]:
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            ax.set_title('#' + str(k))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
            ax.axis('off')
            
            k += 1
            start_temp += nbeam * nscan
        
        count += 1
    axes[1, 7].axis('off')
    axes[2, 6].axis('off')
    axes[2, 7].axis('off')
    axes[3, 6].axis('off')
    axes[3, 7].axis('off')
    axes[4, 5].axis('off')
    axes[4, 6].axis('off')
    axes[4, 7].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
else:
    fig, axes = plt.subplots(nrows=4, ncols=8)
    for ax in axes.flat:
        if k < 21 and count != 5 and count != 6 and count != 7 and count != 23 and count < 26:
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan, :].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            ax.set_title('#' + str(k))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            k += 1
            start_temp += nbeam * nscan
        
        count += 1
    axes[0, 5].axis('off')
    axes[0, 6].axis('off')
    axes[0, 7].axis('off')
    axes[2, 7].axis('off')
    axes[3, 1].axis('off')
    axes[3, 2].axis('off')
    axes[3, 3].axis('off')
    axes[3, 4].axis('off')
    axes[3, 5].axis('off')
    axes[3, 6].axis('off')
    axes[3, 7].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()