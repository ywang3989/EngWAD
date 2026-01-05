from utlis import *
from models import *
import time
from openpyxl import load_workbook
from scipy.stats import zscore
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Hyperparameters
date = '0710_A_9_7_6'
OOD_sample = '_OOD_15_16'
model_selection = 'FFT'
wasser_pattern = 'avg'
feat_selection = 'b'
results_file_name = 'results_' + model_selection + '.xlsx'

depth = 274  # 320 for 1128, 344 for 0805 & 0806, 267 for 0813 & 0821 & 0823, 251 for 1122, 274 for 0521 & 0710

only_test = False
plot_contour = True
image_show = False

parts = date.split('_')
nums = [int(x) for x in parts[2:]]
OOD_parts = OOD_sample.split('_')
OOD_nums = [int(x) for x in OOD_parts[2:]]
file_path = os.getcwd()
file_path = file_path.replace('Codes', 'UT Results')
file_path_suffix = 'TransAE_' + wasser_pattern + '_' + str(nums[0]) + '_' + str(nums[1]) + '_'+ str(nums[2]) + OOD_sample
file_path = os.path.join(file_path, model_selection + '\\' + file_path_suffix)

# Cropping information
# 0710_A:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36]
widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29]
nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

# Get data & normalization
current_path = os.getcwd()
training_file_path = os.path.join(current_path, 'data\\ascan_train_' + date + '.csv')
testing_file_path = os.path.join(current_path, 'data\\ascan_test_' + date + '.csv')
# whole_file_path = os.path.join(current_path, 'data\\ascan_all_' + date + '.csv')
whole_file_path = os.path.join(current_path, 'data\\ascan_all_0710_A.csv')

X_train = np.genfromtxt(training_file_path, delimiter=',').astype('f4')
if only_test:
    X_test = np.genfromtxt(testing_file_path, delimiter=',').astype('f4')
else:
    X_test = np.genfromtxt(whole_file_path, delimiter=',').astype('f4')
X_train = X_train[:, 0:depth]
X_test = X_test[:, 0:depth]

X_train = numpy_per_series_zscore(X_train)
X_test = numpy_per_series_zscore(X_test)

os.makedirs(file_path, exist_ok=True)
time_start = time.perf_counter()

anomaly_score = get_fft_max_amplitude(X_test, seg_start=150, seg_end=200, L=1000)

anomaly_score_min = np.min(anomaly_score, axis=0, keepdims=True)
anomaly_score_max = np.max(anomaly_score, axis=0, keepdims=True)
anomaly_score_norm = 2 * (anomaly_score - anomaly_score_min) / (anomaly_score_max - anomaly_score_min + 1e-8) - 1
anomaly_scores = np.exp(anomaly_score_norm)

np.savetxt(file_path + '\\anomaly_score_TransAE_' + wasser_pattern + '_' + date + '_' + feat_selection + '.csv', anomaly_scores, delimiter = ',')



# Ploting
k = 0
count = 0
start_temp = 0
image_ratio = 1
value_min = np.min(anomaly_scores)
value_max = np.max(anomaly_scores)
plt.style.use('classic')
if date[:4] == '0710':
    sample_num = 20
    row_num = 4
    col_num = 5
    nums_sorted = sorted(nums[:2])
    valid_PB_indices = [i+9 for i in nums_sorted]
    valid_PB_indices_plot = [i-1 for i in nums_sorted]
    aug_OOD_indices_plot = [i-11 for i in OOD_nums]
    test_KB_indices = [i-11 for i in range(11, 21) if i not in OOD_nums]
    train_index_start_num_1 = nums_sorted[0] + 9
    train_index_end_num_1 = nums_sorted[0] + 10
    train_index_start_num_2 = nums_sorted[1] + 9
    train_index_end_num_2 = nums_sorted[1] + 10        
    skip_list = []

    if plot_contour:
        train_index_start_1 = sum([(nbeam_U[i]-nbeam_L[i])*(nscan_U[i]-nscan_L[i]) for i in range(train_index_start_num_1)])
        train_index_end_1 = sum([(nbeam_U[i]-nbeam_L[i])*(nscan_U[i]-nscan_L[i]) for i in range(train_index_end_num_1)])
        train_index_start_2 = sum([(nbeam_U[i]-nbeam_L[i])*(nscan_U[i]-nscan_L[i]) for i in range(train_index_start_num_2)])
        train_index_end_2 = sum([(nbeam_U[i]-nbeam_L[i])*(nscan_U[i]-nscan_L[i]) for i in range(train_index_end_num_2)])
        train_scores_1 = anomaly_scores[train_index_start_1:train_index_end_1]
        train_scores_2 = anomaly_scores[train_index_start_2:train_index_end_2]
        train_scores = np.concatenate((train_scores_1, train_scores_2))
        
        # Training score histogram
        quantile = 95
        pixel_thre = np.percentile(train_scores, quantile)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(train_scores, bins=50, facecolor='none', hatch='///', linewidth=1.2, label='Training Score Histogram')
        ax.axvline(pixel_thre, color='red', linestyle='--', linewidth=2, label=rf'Pixel-Level Threshold $t_s={pixel_thre:.3f}$')
        ax.set_xlabel('Training Pixel-Level Score')
        ax.set_ylabel('Frequency')
        ax.legend()
        image_path_1 = file_path + '\\pixel_score_histogram_' + feat_selection + '.png'
        plt.savefig(image_path_1, bbox_inches='tight')
        if image_show:
            plt.show()

        # Visualization
        start_temp, k = 0, 0
        sample_anomaly_indices = np.zeros(sample_num)
        fig, axes = plt.subplots(figsize=(24, 12), nrows=row_num, ncols=col_num)
        for ax in axes.flat:
            if k < sample_num and count not in skip_list:
                if k < 10:
                    row = 2 + (k // 5)
                    col = k % 5
                else:
                    j   = k - 10
                    row = j // 5
                    col = j % 5
                ax = axes[row, col]

                nbeam = nbeam_U[k] - nbeam_L[k]
                nscan = nscan_U[k] - nscan_L[k]
                score = anomaly_scores[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
                im = ax.imshow(score, vmin=value_min, vmax=value_max)

                x = np.arange(score.shape[1])
                y = np.arange(score.shape[0])
                X, Y = np.meshgrid(x, y)
                contour = ax.contour(X, Y, score, levels=[pixel_thre], colors="black", linewidths=0.5)
                ax.clabel(contour, inline=True, fontsize=8)
                
                anomaly_area = np.sum((score > pixel_thre).astype(int))
                sample_anomaly_index = anomaly_area / (nbeam * nscan)  # iou
                sample_anomaly_indices[k] = sample_anomaly_index

                # score = np.maximum(score, pixel_thre)
                # score = score[~np.all(score == pixel_thre)]
                # if not np.any(score):
                #     score_avg, score_std = 0, 0
                # else:
                #     score_avg, score_std = np.mean(score), np.std(score)

                title_idx = row * 5 + col
                title_label = (
                    f'#{title_idx+1} Anomaly Index: {sample_anomaly_index:.3f}\n'
                    f'Avg Score: {np.mean(score):.4f}'
                )
                ax.set_title(title_label, fontsize=10)
                # ax.set_title('#' + str(k+1))
                ax.axis('off')

                x_left, x_right = ax.get_xlim()
                y_low, y_high = ax.get_ylim()
                ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * image_ratio)

                k += 1
                start_temp += nbeam * nscan

            count += 1
        
        if date == '0710_A_wo2&18':
            axes[0, 1].axis('off')
            axes[3, 2].axis('off')

        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.ax.hlines(pixel_thre, xmin=0, xmax=1, color='k', linewidth=1)
        sup_title_label = (f'Pixel-Level Threshold: {pixel_thre:.3f}')
        fig.suptitle(sup_title_label)
        image_path_2 = file_path + '\\anomaly_map_' + str(round(pixel_thre, 3)) + '_' + feat_selection + '.png'
        plt.savefig(image_path_2, bbox_inches='tight')
        if image_show:
            plt.show()

        # Sample level chart
        valid_PB_anomaly_indices = []
        start_temp = sum([(nbeam_U[i]-nbeam_L[i])*(nscan_U[i]-nscan_L[i]) for i in range(10)])
        for k in range(10, 20):
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            score = anomaly_scores[start_temp:start_temp+nbeam*nscan].reshape((nbeam, nscan))
            anomaly_area = np.sum((score > pixel_thre).astype(int))
            sample_anomaly_index = anomaly_area / (nbeam * nscan)

            if k in valid_PB_indices:
                valid_PB_anomaly_indices.append(sample_anomaly_index)

            start_temp += nbeam * nscan

        valid_PB_anomaly_indices = np.array(valid_PB_anomaly_indices)
        x_bar = 0.01 * (100 - quantile)
        factor = 3 / np.sqrt(valid_PB_anomaly_indices.shape[0])
        mu = np.mean(valid_PB_anomaly_indices)
        sigma = valid_PB_anomaly_indices.std(ddof=1)
        sample_thre = max([mu, x_bar]) + factor * sigma

        sample_anomaly_indices_PB = sample_anomaly_indices[10:]
        sample_anomaly_indices_KB = sample_anomaly_indices[:10]
        sample_anomaly_indices_reordered = np.concatenate((sample_anomaly_indices_PB, sample_anomaly_indices_KB))
        x = np.arange(1, sample_anomaly_indices_reordered.shape[0]+1)
        x_PB = np.arange(1, 11)
        x_KB = np.arange(11, 21)
        fig, ax = plt.subplots(figsize=(10, 6))
        # fig, ax = plt.subplots(figsize=(20, 12))
        ax.plot(x, sample_anomaly_indices_reordered, color='blue', linestyle='--', label='Sample-Level Anomlay Index')
        ax.axhline(sample_thre, color='red', linestyle='-', label=rf'Sample-Level Threshold $t_p={sample_thre:.3f}$')
        # ax.scatter(x_PB[nums[-1]-1], sample_anomaly_indices_PB[nums[-1]-1], color='green', marker='^', s=80, label='Pristine for Testing', zorder=5)
        # ax.scatter(x_PB[valid_PB_indices_plot], sample_anomaly_indices_PB[valid_PB_indices_plot], color='cyan', marker='^', s=80, label='Pristine for Thresholding', zorder=5)
        normal_mask_PB = sample_anomaly_indices_PB < sample_thre
        ax.scatter(x_PB[normal_mask_PB], sample_anomaly_indices_PB[normal_mask_PB], color='blue', marker='^', s=80, label='Pristine Not Signaled', zorder=5)
        anomaly_mask_PB = sample_anomaly_indices_PB >= sample_thre
        ax.scatter(x_PB[anomaly_mask_PB], sample_anomaly_indices_PB[anomaly_mask_PB], color='red', marker='^', s=80, label='Pristine Signaled', zorder=5)
        # ax.scatter(x_KB[aug_OOD_indices_plot], sample_anomaly_indices_KB[aug_OOD_indices_plot], color='blue', marker='o', s=80, label='Weak Adhesion for OOD Aug.', zorder=5)
        # ax.scatter(x_KB, sample_anomaly_indices_KB, color='green', marker='o', s=70, label='Weak Adhesion for Testing', zorder=4)
        normal_mask_KB = sample_anomaly_indices_KB < sample_thre
        ax.scatter(x_KB[normal_mask_KB], sample_anomaly_indices_KB[normal_mask_KB], color='blue', marker='o', s=80, label='Weak Adhesion Not Signaled', zorder=5)
        anomaly_mask_KB = sample_anomaly_indices_KB >= sample_thre
        ax.scatter(x_KB[anomaly_mask_KB], sample_anomaly_indices_KB[anomaly_mask_KB], color='red', marker='o', s=80, label='Weak Adhesion Signaled', zorder=5)
        ax.set_xlabel('Sample #')
        ax.set_ylabel('Sample-Level Anomlay Index')
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(-0.1, 1.6)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
        ax.legend(loc='upper left', fontsize=13)
        image_path_3 = file_path + '\\sample_level_thresholding_' + feat_selection + '.png'
        plt.savefig(image_path_3, bbox_inches='tight')
        if image_show:
            plt.show()

        # Quantitative metrics
        y_pred = np.concatenate((sample_anomaly_indices_PB[[nums[-1]-1]], sample_anomaly_indices_KB[test_KB_indices]), axis=0)
        y_true = np.array([0, 1, 1, 1, 1, 1, 1, 1, 1])
        y_pred_label = (y_pred >= sample_thre).astype(int)

        acc  = accuracy_score(y_true, y_pred_label)
        prec = precision_score(y_true, y_pred_label, zero_division=0)
        rec  = recall_score(y_true, y_pred_label, zero_division=0)
        f1   = f1_score(y_true, y_pred_label, zero_division=0)
        print(f"Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        time_end_ = time.perf_counter()
        total_time = time_end_ - time_start
        print(f'In total {total_time:.6f} seconds')

        if feat_selection == 'b':
            wb = load_workbook(results_file_name)
            ws = wb["Sheet1"] 
            ws.append([date, OOD_sample, acc, prec, rec, f1, total_time])
            wb.save(results_file_name)



        

