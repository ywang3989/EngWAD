import numpy as np
import os
import os.path
import matplotlib.pyplot as plt

date = '0710_A'
# intersec_l = 160
# intersec_u = 230
L = 1000

if date == '0805':
    nbeam_L = [16, 10, 12, 13, 15, 15, 15, 12, 14, 15, 14, 15, 13,  9,  6,  6, 10, 10,  9, 11, 10]
    nbeam_U = [48, 40, 41, 43, 41, 45, 44, 40, 40, 43, 43, 40, 41, 46, 44, 48, 43, 41, 44, 43, 46]
    nscan_L = [22, 20, 24, 23, 22, 24, 22, 22, 22, 22, 23, 20, 20, 20, 17, 20, 17, 17, 17, 17, 20]
    nscan_U = [43, 44, 44, 46, 46, 46, 44, 46, 45, 44, 45, 38, 40, 46, 45, 49, 46, 47, 44, 45, 43]
elif date == '0813':
    nbeam_L = [10, 10, 13, 17, 10, 20, 16, 19, 12, 10, 10,  7, 13, 14,  9, 12, 10, 11,  8, 10, 12, 10,  8, 10,  7,  9, 10, 10,  5,  6,  4,  7,  6,  5,  5,  7]
    nbeam_U = [43, 39, 43, 44, 43, 45, 42, 45, 41, 41, 43, 40, 40, 44, 43, 42, 43, 41, 42, 44, 46, 45, 43, 43, 42, 44, 41, 43, 48, 48, 48, 45, 41, 44, 43, 45]
    nscan_L = [22, 18, 23, 16, 18, 15, 18, 26, 25, 17, 23, 24, 23, 20, 20, 20, 20, 30, 20, 32, 20, 20, 20, 20, 24, 20, 19, 19, 20, 15, 20, 18, 18, 17, 15, 14]
    nscan_U = [43, 41, 46, 38, 44, 41, 43, 51, 47, 43, 47, 50, 46, 44, 45, 44, 43, 51, 46, 56, 48, 47, 48, 48, 51, 48, 46, 49, 53, 51, 55, 51, 51, 48, 46, 43]
elif date == '0821':
    nbeam_L = [10, 10, 13, 17, 10, 20, 16, 19, 12, 10,  8, 10,  9, 10, 10, 12, 10, 10,  7, 13, 14,  9, 12, 10, 11,  8, 10,  7, 13, 10, 10, 12,  5,  6,  4,  7,  6,  5,  5,  7]
    nbeam_U = [43, 39, 43, 44, 43, 45, 42, 45, 46, 45, 43, 43, 44, 41, 43, 41, 41, 43, 40, 40, 44, 43, 42, 43, 41, 42, 44, 42, 44, 43, 39, 42, 48, 48, 48, 45, 41, 44, 43, 45]
    nscan_L = [22, 18, 23, 16, 18, 15, 18, 26, 20, 20, 20, 20, 20, 19, 19, 25, 17, 23, 24, 23, 20, 20, 20, 20, 30, 20, 32, 24, 20, 25, 20, 25, 20, 15, 20, 18, 18, 17, 15, 14]
    nscan_U = [43, 41, 46, 38, 44, 41, 43, 51, 48, 47, 48, 48, 48, 46, 49, 47, 43, 47, 50, 46, 44, 45, 44, 43, 51, 46, 56, 51, 46, 51, 46, 51, 53, 51, 55, 51, 51, 48, 46, 43]
elif date == '0823':
    nbeam_L = [49, 56, 49, 48, 49, 48, 47, 43, 51, 49, 48, 48, 50, 50, 45, 49, 46, 48, 47, 46, 44, 43, 43, 47, 45, 44, 48, 47, 50, 50, 51, 47, 47, 41, 43, 50, 47, 46, 41, 47]
    nbeam_U = [75, 81, 73, 72, 75, 70, 73, 67, 78, 77, 73, 73, 76, 73, 70, 72, 70, 73, 71, 64, 65, 66, 64, 72, 70, 72, 73, 75, 76, 77, 77, 68, 73, 71, 73, 77, 72, 72, 69, 75]
    nscan_L = [33, 31, 31, 32, 42, 43, 38, 33, 35, 34, 35, 33, 32, 33, 31, 33, 27, 45, 29, 31, 34, 34, 30, 32, 30, 32, 31, 39, 37, 41, 34, 32, 23, 29, 30, 40, 29, 35, 31, 30]
    nscan_U = [54, 51, 52, 53, 64, 67, 59, 56, 56, 56, 59, 56, 53, 51, 49, 52, 45, 64, 48, 49, 52, 54, 50, 53, 50, 52, 52, 58, 56, 60, 53, 51, 48, 58, 53, 67, 53, 55, 52, 49]
elif date == '1122':
    nbeam_L = [12, 12,  6, 12, 12, 19, 16,  8, 21, 11, 13, 11, 12, 16, 14, 15, 21, 16, 10, 14, 19, 17, 15, 14, 14, 15, 13, 13, 12, 11, 12, 15, 16, 17, 13, 15, 11, 16, 17, 14, 14, 16, 17, 13, 16, 16, 14, 15, 15, 17, 15, 14, 14, 13, 12, 16, 11,  6,  7,  9, 10,  8,  8,  7]
    nbeam_U = [39, 41, 34, 37, 39, 42, 42, 35, 46, 38, 39, 40, 37, 41, 39, 40, 44, 42, 43, 42, 41, 41, 42, 42, 41, 42, 40, 40, 39, 38, 39, 46, 41, 42, 38, 40, 36, 41, 45, 45, 42, 42, 46, 44, 43, 45, 41, 42, 42, 41, 41, 41, 40, 40, 39, 41, 47, 47, 49, 45, 42, 42, 43, 41]
    nscan_L = [23, 15, 19, 19, 22, 19, 24, 17, 26, 20, 22, 20, 22, 20, 23, 20, 20, 20, 18, 21, 23, 22, 22, 23, 26, 22, 22, 22, 38, 25, 22, 22, 26, 22, 26, 23, 30, 25, 37, 24, 23, 37, 21, 20, 20, 19, 18, 19, 19, 18, 19, 25, 22, 28, 21, 22, 19, 15, 15, 13, 15, 16, 12, 14]
    nscan_U = [46, 38, 40, 40, 45, 40, 49, 40, 50, 45, 47, 48, 47, 45, 46, 43, 41, 43, 49, 47, 45, 42, 48, 50, 53, 49, 48, 49, 69, 50, 49, 50, 51, 45, 48, 48, 53, 47, 64, 52, 47, 58, 46, 44, 42, 43, 43, 43, 42, 40, 41, 48, 43, 50, 45, 44, 51, 50, 51, 43, 44, 45, 40, 44]
elif date == '0521':
    nbeam_L = [48, 50, 47, 48, 50, 57, 48, 49, 48, 55, 53, 50, 48, 49, 52, 48, 49, 54, 52, 54, 51, 50, 52, 48, 50, 53, 58, 47, 53, 50, 52, 44, 47, 50, 48, 50, 43, 45, 52, 48, 50, 56, 49, 48, 52, 44, 49, 50, 51, 55, 57, 54, 57, 52, 50, 47, 50, 46, 50, 48, 52, 43]
    heights = [23, 22, 22, 20, 23, 23, 24, 23, 24, 26, 19, 21, 22, 21, 26, 24, 28, 25, 20, 20, 24, 24, 23, 23, 26, 24, 22, 23, 25, 24, 22, 24, 21, 20, 26, 25, 27, 28, 22, 24, 27, 26, 25, 27, 25, 26, 25, 21, 27, 20, 22, 19, 18, 27, 28, 30, 25, 22, 25, 26, 27, 28]
    nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
    nscan_L = [37, 35, 33, 33, 38, 40, 45, 40, 42, 42, 42, 42, 43, 43, 44, 35, 42, 55, 33, 39, 40, 39, 44, 40, 41, 56, 42, 35, 40, 29, 40, 42, 40, 35, 38, 36, 39, 40, 45, 55, 39, 39, 38, 38, 45, 40, 40, 51, 49, 39, 37, 42, 59, 39, 32, 37, 39, 50, 36, 40, 47, 38]
    widths  = [24, 22, 20, 24, 24, 24, 24, 20, 23, 22, 23, 24, 24, 26, 24, 20, 26, 26, 18, 20, 23, 24, 23, 26, 22, 24, 21, 18, 25, 24, 21, 20, 20, 18, 21, 19, 26, 24, 17, 18, 23, 20, 22, 23, 23, 19, 23, 19, 22, 19, 23, 23, 21, 30, 26, 23, 27, 27, 24, 23, 23, 25]
    nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]
elif date == '0710_AB':
    nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43, 52, 53, 51, 54, 53, 52, 47, 52, 54, 53]
    heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26, 26, 26, 27, 27, 25, 26, 26, 26, 26, 27]
    nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
    nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36, 46, 47, 38, 40, 41, 42, 39, 58, 59, 36]
    widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29, 28, 23, 27, 25, 24, 25, 28, 29, 28, 28]
    nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]
elif date == '0710_A':
    nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
    heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
    nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
    nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36]
    widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29]
    nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]
elif date == '0710_A_wo2&18':
    nbeam_L = [48, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 46, 43]
    heights = [22, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 27, 26]
    nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
    nscan_L = [46, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 43, 36]
    widths  = [23, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 29]
    nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]
elif date == '0802':
    nbeam_L = [48, 50, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
    heights = [22, 28, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
    nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
    nscan_L = [46, 40, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 38, 43, 36]
    widths  = [23, 20, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 22, 27, 29]
    nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

current_path = os.getcwd()
ascan_all_path = os.path.join(current_path, 'data\\ascan_all_' + date + '.csv')
ascan_all = (np.genfromtxt(ascan_all_path, delimiter = ',') - 127) / 127

if date == '1128':
    ascan_all_path = os.path.join(current_path, 'data\\whole.csv')
    ascan_all = np.genfromtxt(ascan_all_path, delimiter = ',')

A = np.zeros((ascan_all.shape[0], 1))
for i in range(ascan_all.shape[0]):
    ascan = ascan_all[i, :]
    pos = np.argmax(ascan[150:200])
    intersec_l = pos + 140  # +150-10
    intersec_u = int(np.min([pos + 200, 250]))  # +150+50
    fft_signal = np.abs(np.fft.fft(ascan[intersec_l:intersec_u], n=L))
    A[i, :] = np.max(fft_signal[0:int(L/2)])

# Plot
k = 0
count = 0
start_temp = 0
ratio = 1
value_min = np.min(A)
value_max = np.max(A)
plt.style.use('classic')
if date == '1128':  # Without mouse encoders
    width = 30
    A[5400:8100], A[8100:10800] = A[8100:10800], A[5400:8100].copy()
    fig, axes = plt.subplots(nrows = 3, ncols = 6)
    for ax in axes.flat:
        if k < 15 and count != 3 and count != 4 and count != 5:
            score = A[k*width*width:(k+1)*width*width].reshape((width, width))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)
            ax.set_title('#' + str(k))

            k += 1
        count += 1
    axes[0, 3].axis('off')
    axes[0, 4].axis('off')
    axes[0, 5].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0710_A' or date == '0802':
    fig, axes = plt.subplots(nrows=4, ncols=5)

    for ax in axes.flat:
        if k < 20:
            if k < 10:
                # first 10 → 3rd and 4th rows (zero-based 2 and 3)
                row = 2 + (k // 5)   # k=0..4 → 2, k=5..9 → 3
                col = k % 5
            else:
                # next 10 → 1st and 2nd rows (zero-based 0 and 1)
                j   = k - 10
                row = j // 5         # j=0..4 → 0, j=5..9 → 1
                col = j % 5
            ax = axes[row, col]

            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            title_idx = row * 5 + col
            ax.set_title('#' + str(title_idx+1))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            k += 1
            start_temp += nbeam * nscan
            
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0710_A_wo2&18':
    fig, axes = plt.subplots(nrows=4, ncols=5)
    for ax in axes.flat:
        if k < 18 and count not in [1, 17]:
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            ax.set_title('#' + str(count+1))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            # if k == 0:  in [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19]
            #     start_temp = nbeam * nscan + (nbeam_U[1]-nbeam_L[1]) * (nscan_U[1]-nscan_L[1])
            #     k += 2
            # elif k in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
            #     start_temp += nbeam * nscan
            #     k += 1
            # elif k == 16:
            #     start_temp = start_temp + nbeam * nscan + (nbeam_U[17]-nbeam_L[17]) * (nscan_U[17]-nscan_L[17])
            #     k += 2
            # elif k in [18, 19]:
            #     start_temp += nbeam * nscan
            #     k += 1

            k += 1
            start_temp += nbeam * nscan

        count += 1
    axes[0, 1].axis('off')
    axes[3, 2].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0710_AB':
    fig, axes = plt.subplots(nrows=3, ncols=10)
    for ax in axes.flat:
        if k < 30:
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = A[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            im = ax.imshow(score, vmin=value_min, vmax=value_max)

            ax.set_title('#' + str(k+1))
            x_left, x_right = ax.get_xlim()
            y_low, y_high = ax.get_ylim()
            ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

            k += 1
            start_temp += nbeam * nscan
        
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0521':
    fig, axes = plt.subplots(nrows=9, ncols=9)
    for ax in axes.flat:
        if k < 62 and count not in [6, 7, 8, 15, 16, 17, 24, 25, 26, 35, 44, 53, 60, 61, 62, 68, 69, 70, 71]:
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
    axes[0, 6].axis('off')
    axes[0, 7].axis('off')
    axes[0, 8].axis('off')
    axes[1, 6].axis('off')
    axes[1, 7].axis('off')
    axes[1, 8].axis('off')
    axes[2, 6].axis('off')
    axes[2, 7].axis('off')
    axes[2, 8].axis('off')
    axes[3, 8].axis('off')
    axes[4, 8].axis('off')
    axes[5, 8].axis('off')
    axes[6, 6].axis('off')
    axes[6, 7].axis('off')
    axes[6, 8].axis('off')
    axes[7, 5].axis('off')
    axes[7, 6].axis('off')
    axes[7, 7].axis('off')
    axes[7, 8].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '1122':
    fig, axes = plt.subplots(nrows=9, ncols=8)
    for ax in axes.flat:
        if k < 64 and count not in [14, 15, 22, 23, 55, 61, 62, 63]:
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
        if k < 36 and count != 28 and count != 29 and count != 38 and count != 39:
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
    axes[2, 8].axis('off')
    axes[2, 9].axis('off')
    axes[3, 8].axis('off')
    axes[3, 9].axis('off')
    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.show()
elif date == '0821' or date == '0823':
    fig, axes = plt.subplots(nrows=6, ncols=8)
    for ax in axes.flat:
        if k < 40 and count not in [15, 22, 23, 30, 31, 37, 38, 39]:
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

