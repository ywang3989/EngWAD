from utlis import *
from models import *
import time
from openpyxl import load_workbook
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Hyperparameters
date = '0710_A_3_8_10'
OOD_sample = '_OOD_14_16'
model_selection = 'EngAD'
results_file_name = 'results_' + model_selection + '.xlsx'

depth = 274  # 320 for 1128, 344 for 0805 & 0806, 267 for 0813 & 0821 & 0823, 251 for 1122, 274 for 0521 & 0710
plot_value_max = 0.15

batch_size = 512
r = 4
h_dim = 128
p_dim = 128

feat_selection = 'b' # 'h' or 'z' or 'b'
wasser_pattern = 'avg'
tau = 0.1
keep_ratio = 0.999

whether_coreset = False
coreset_ratio = 0.9999

only_test = False
read_saved_file = False
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
X_train = torch.from_numpy(X_train[:, 0:depth]).view(-1, 1, depth).to(device)
X_test = torch.from_numpy(X_test[:, 0:depth]).view(-1, 1, depth).to(device)

X_train = per_series_zscore(X_train).float()
X_test = per_series_zscore(X_test).float()

if not read_saved_file:
    os.makedirs(file_path, exist_ok=True)
    # Initialization
    # model = ContraAE(c_in=1, r=r, h_dim=h_dim, p_dim=p_dim).to(device)
    model = TransAE(c_in=1, r=r, h_dim=h_dim, p_dim=p_dim).to(device)
    state_file = torch.load(os.path.join('model_para', f'{wasser_pattern}\{model_selection}\{model_selection}_{wasser_pattern}_{date+OOD_sample}.pth'), map_location=device)
    model.load_state_dict(state_file, strict=True)
    model.eval()
    encoder = model.ae.encoder
    decoder = model.ae.decoder

    # state_encoder = torch.load(os.path.join('model_para', f'ContraAE_Encoder_{date}_tau_{tau}.pth'), map_location=device)
    # encoder.load_state_dict(state_encoder, strict=True)
    # encoder.eval()

    # state_decoder = torch.load(os.path.join('model_para', f'ContraAE_Decoder_{date}_tau_{tau}.pth'), map_location=device)
    # decoder.load_state_dict(state_decoder, strict=True)
    # decoder.eval()

    with torch.no_grad():
        hidden_features_all = encoder(X_test)

    anomaly_score = np.zeros((X_test.shape[0], 2))
    
    # hidden_features_training = encoder(X_train)
    hidden_features_training_z, hidden_features_training_h, _ = model.encode_to_proj(X_train)

    # Coreset for training features
    if whether_coreset:
        # h
        hidden_features_training_h_ = hidden_features_training_h.reshape(X_train.shape[0], -1)
        coreset_sampler = ApproximateGreedyCoresetSampler(percentage=coreset_ratio, device=device)
        coreset_sampling_flag = True
        while coreset_sampling_flag:
            coreset_indices = coreset_sampler._compute_greedy_coreset_indices(hidden_features_training_h_)
            print(len(set(coreset_indices)), len(coreset_indices))
            if len(set(coreset_indices)) >= 0.9 * len(coreset_indices):
                coreset_sampling_flag = False
        np.savetxt(file_path + '\\coreset_indices_' + date + '.csv', coreset_indices, delimiter = ',')
        coreset_indices = list(set(coreset_indices))
        hidden_features_training_selected_h = hidden_features_training_h[coreset_indices]
        coreset_image_path = file_path + '\\coreset_' + date + '_' + str(coreset_ratio) + '.png'
        coreset_pca_visualization(hidden_features_training_h_.detach().cpu().numpy(), coreset_indices, coreset_image_path)
        # z
        hidden_features_training_z_ = hidden_features_training_z.reshape(X_train.shape[0], -1)
        coreset_sampler = ApproximateGreedyCoresetSampler(percentage=coreset_ratio, device=device)
        coreset_sampling_flag = True
        while coreset_sampling_flag:
            coreset_indices = coreset_sampler._compute_greedy_coreset_indices(hidden_features_training_z_)
            print(len(set(coreset_indices)), len(coreset_indices))
            if len(set(coreset_indices)) >= 0.9 * len(coreset_indices):
                coreset_sampling_flag = False
        np.savetxt(file_path + '\\coreset_indices_' + date + '.csv', coreset_indices, delimiter = ',')
        coreset_indices = list(set(coreset_indices))
        hidden_features_training_selected_z = hidden_features_training_z[coreset_indices]
        coreset_image_path = file_path + '\\coreset_' + date + '_' + str(coreset_ratio) + '.png'
        coreset_pca_visualization(hidden_features_training_z_.detach().cpu().numpy(), coreset_indices, coreset_image_path)
    else:
        hidden_features_training_selected_h = hidden_features_training_h
        hidden_features_training_selected_z = hidden_features_training_z

    image_path_0_h = file_path + '\\t-SNE_h.png'
    N_selected_h = hidden_features_training_selected_h.shape[0]
    hidden_features_training_temp_h = hidden_features_training_selected_h.view(N_selected_h, -1).cpu().detach().numpy()
    keep_idx_h, _ = filter_by_iforest(hidden_features_training_temp_h, keep_ratio=keep_ratio)
    visualize_tsne_before_after(hidden_features_training_temp_h, keep_idx_h, image_path_0_h)
    hidden_features_training_selected_h = hidden_features_training_selected_h[keep_idx_h]

    image_path_0_z = file_path + '\\t-SNE_z.png'
    N_selected_z = hidden_features_training_selected_z.shape[0]
    hidden_features_training_temp_z = hidden_features_training_selected_z.view(N_selected_z, -1).cpu().detach().numpy()
    keep_idx_z, _ = filter_by_iforest(hidden_features_training_temp_z, keep_ratio=keep_ratio)
    visualize_tsne_before_after(hidden_features_training_temp_z, keep_idx_z, image_path_0_z)
    hidden_features_training_selected_z = hidden_features_training_selected_z[keep_idx_z]

    # Memory processing
    time_start = time.perf_counter()
    batch_start = torch.arange(0, X_test.shape[0], batch_size)
    print(f'In toatl {batch_start.shape[0]} batches')
    for start in batch_start:
        X = X_test[start:start+batch_size, :]
        # anomaly_scores_l2[start:start+batch_size] = torch.norm(X - X_recon, dim=-1).view(-1).detach().cpu().numpy()
        batch_time_start = time.perf_counter()
        # hidden_memory, recon_Frechet, recon_Euclidean = memory_processing_fast_AE(encoder, decoder, hidden_features_training_selected, X, 0, 274, r, R=1)
        anomaly_score_embedding_h = memory_processing_fast_ContraAE(model, hidden_features_training_selected_h, 
                                                                    X, R=1, feat_selection='h')
        anomaly_score_embedding_z = memory_processing_fast_ContraAE(model, hidden_features_training_selected_z, 
                                                                    X, R=1, feat_selection='z')
        
        anomaly_score_embedding = np.column_stack((anomaly_score_embedding_h, anomaly_score_embedding_z))
        batch_time_end = time.perf_counter()
        anomaly_score[start:start+batch_size] = anomaly_score_embedding
        print(f'Finished # {int(start/batch_size)+1} batch! ({batch_time_end - batch_time_start:.6f} seconds)')
    time_end = time.perf_counter()
    print(f'In total {time_end - time_start:.6f} seconds')
    
    
    if feat_selection == 'h':
        anomaly_scores = np.exp(anomaly_score[:, 0])
    elif feat_selection == 'z':
        anomaly_scores = np.exp(anomaly_score[:, 1])
    elif feat_selection == 'b':
        anomaly_score_min = np.min(anomaly_score, axis=0, keepdims=True)
        anomaly_score_max = np.max(anomaly_score, axis=0, keepdims=True)
        anomaly_score_norm = 2 * (anomaly_score - anomaly_score_min) / (anomaly_score_max - anomaly_score_min) - 1
        anomaly_scores = np.exp(np.max(anomaly_score_norm, axis=1))
    else:
        raise ValueError("Invalid input")
    
    np.savetxt(file_path + '\\anomaly_score_TransAE_' + wasser_pattern + '_' + date + '_' + feat_selection + '.csv', anomaly_scores, delimiter = ',')
else:
    # Reading anomaly score file
    anomaly_scores_embedding_file_name = file_path + '\\anomaly_score_TransAE_' + wasser_pattern + '_' + date + '_' + feat_selection + '.csv'
    anomaly_scores = np.genfromtxt(anomaly_scores_embedding_file_name, delimiter=',').astype('f4')


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



        

