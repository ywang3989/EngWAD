from utlis import *
from models import *

# Hyperparameters
testing_sample = '4_10_7'
OOD_sample = '11_17'
feat_selection = 'b' # 'h' or 'z' or 'b'
wasser_pattern = 'avg'

# parts = date.split('_')
# nums = [int(x) for x in parts[2:]]
# OOD_parts = OOD_sample.split('_')
# OOD_nums = [int(x) for x in OOD_parts[2:]]
# file_path = os.getcwd()
# file_path = file_path.replace('Codes', 'UT Results')
# file_path_suffix = 'TransAE_' + wasser_pattern + '_' + str(nums[0]) + '_' + str(nums[1]) + '_'+ str(nums[2]) + OOD_sample

# Cropping information
# 0710_A:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36]
widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29]
nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

sample_sizes = [(nbeam_U[k] - nbeam_L[k]) * (nscan_U[k] - nscan_L[k]) for k in range(20)]
sample_starts = np.insert(np.cumsum(sample_sizes), 0, 0)  # [0, size0, size0+size1, ...]
target_samples = [13, 19, 0, 6, 16, 2, 5]
sample_labels = [4, 10, 11, 17, 7, 13, 16]
model_selections = ['EngWAD', 'WAD', 'EngAD', 'AD', 'FFT']
pixel_threshold = [0.745, 0.706, 1.104, 0.861, 0.874]

fig, axes = plt.subplots(
    len(model_selections), 
    len(target_samples), 
    figsize=(20, 12),
    gridspec_kw={
        'wspace': 0.1, # Horizontal gap (width)
        'hspace': 0.1  # Vertical gap (height)
    }
)
value_min = np.exp(-1)
value_max = np.exp(1)
image_ratio = 1
plt.style.use('classic')
for i, model_selection in enumerate(model_selections):
    file_name = f'anomaly_score_TransAE_{wasser_pattern}_0710_A_{testing_sample}_{feat_selection}.csv'
    file_path = f'results_file\\{model_selection}\\TransAE_{wasser_pattern}_{testing_sample}_OOD_{OOD_sample}\\'
    full_path = os.path.join(file_path, file_name)
    anomaly_scores = np.genfromtxt(full_path, delimiter=',').astype('f4')

    # Get current model threshold
    pixel_thre = pixel_threshold[i]

    for j, sample_idx in enumerate(target_samples):
        ax = axes[i, j]
        
        # Determine geometry and slice the flat array
        nbeam = nbeam_U[sample_idx] - nbeam_L[sample_idx]
        nscan = nscan_U[sample_idx] - nscan_L[sample_idx]
        start_temp = sample_starts[sample_idx]
        
        # Reshape to 2D map
        score = anomaly_scores[start_temp : start_temp + nbeam * nscan].reshape((nbeam, nscan))
        
        # Plot heatmap
        im = ax.imshow(score, vmin=value_min, vmax=value_max)

        x = np.arange(score.shape[1])
        y = np.arange(score.shape[0])
        X, Y = np.meshgrid(x, y)
        contour = ax.contour(X, Y, score, levels=[pixel_thre], colors="black", linewidths=0.5)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Calculate Subtitle Metrics
        anomaly_area = np.sum((score > pixel_thre).astype(int))
        sample_anomaly_index = anomaly_area / (nbeam * nscan)
        
        # title_label = (
        #     f'Anomaly Index: {sample_anomaly_index:.3f}\n'
        #     f'Avg. Anomaly Score: {np.mean(score):.4f}'
        # )
        # ax.set_title(title_label, fontsize=8)
        
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * image_ratio)

        if i == 0:
            ax.set_xlabel(f"Sample #{sample_labels[j]}", fontsize=12, fontweight='bold')
            ax.xaxis.set_label_position('top') 
        if j == 0:
            ax.set_ylabel(model_selection, fontsize=12, fontweight='bold')
        ax.axis('off')

# Add a colorbar at the end
cbar = fig.colorbar(im, ax=axes.ravel().tolist())

# plt.suptitle(f'Comparative Model Performance (Date: {date})', fontsize=20)
plt.show()
    