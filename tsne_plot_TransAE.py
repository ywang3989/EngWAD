from utlis import *
from models import *
from scipy.stats import gaussian_kde
from torch.utils.data import TensorDataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

date = '0710_A_4_10_7'
OOD_sample = '_OOD_11_17'
depth = 274

model_selection = 'EngWAD'
feat_selection = 'h'
wasser_pattern = 'avg'

r = 4
h_dim = 128
p_dim = 128
tau = 0.1
keep_ratio = 0.95
LATENT_C, LATENT_H = 16, 18
FEATURE_DIM = LATENT_C * LATENT_H


# -------------------- Index Definitions --------------------
parts = date.split('_')
nums = [int(x) for x in parts[2:]]
nums_sorted = sorted(nums[:2])
train_pristine_indices = list(range(10, 20))
for idx in nums:
    train_pristine_indices.remove(idx + 9)
valid_pristine_indices = [i + 9 for i in nums_sorted]
test_pristine_indices  = [nums[-1] + 9]

nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36]
widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29]
nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]
sample_num = len(nbeam_L)


# -------------------- Load & Normalize Signals --------------------
root = os.getcwd().replace('Codes', 'UT Results')
file_path = os.path.join(root, 'B4_0710', '0710_TransAE_'+wasser_pattern, 'TransAE_'+date[7:]+OOD_sample)
os.makedirs(file_path, exist_ok=True)

csv_path = os.path.join(os.getcwd(), 'data', 'ascan_all_0710_A.csv')
X_all_np = np.genfromtxt(csv_path, delimiter=',').astype('f4')
X_all    = torch.from_numpy(X_all_np[:, :depth]).view(-1, 1, depth)
X_all    = per_series_zscore(X_all.float())
dl_all   = DataLoader(TensorDataset(X_all), batch_size=1024, shuffle=False)


# -------------------- Load Trained Encoder --------------------
# model = ContraAE(c_in=1, r=r, h_dim=h_dim, p_dim=p_dim).to(device)
model = TransAE(c_in=1, r=r, h_dim=h_dim, p_dim=p_dim).to(device)
state_file = torch.load(os.path.join('model_para', f'{wasser_pattern}\{model_selection}\{model_selection}_{wasser_pattern}_{date+OOD_sample}.pth'), map_location=device)
model.load_state_dict(state_file, strict=True)
model.eval()

# encoder = model.ae.encoder.to(device)
# state_encoder = torch.load(os.path.join('model_para', f'ContraAE_Encoder_{date}_tau_{tau}.pth'), map_location=device)
# encoder.load_state_dict(state_encoder, strict=True)
# encoder.eval()


# -------------------- Latent Extraction (N×16×18 → N×288) --------------------
z_list = []
with torch.inference_mode():
    for (xb,) in dl_all:
        xb = xb.to(device)
        z, h, _ = model.encode_to_proj(xb)
        if feat_selection == 'h':
            feat = h.cpu()
        elif feat_selection == 'z':
            feat = z.cpu()
        else:
            raise ValueError("Invalid input")
        z_list.append(feat)
z_all = torch.cat(z_list, dim=0)
Z_flat = z_all.reshape(z_all.size(0), -1).numpy().astype('float32')


# -------------------- Split by Geometry --------------------
counts = [(nbeam_U[k] - nbeam_L[k]) * (nscan_U[k] - nscan_L[k]) for k in range(sample_num)]
edges  = np.cumsum([0] + counts)
features_per_sample = [Z_flat[edges[k]:edges[k+1], :] for k in range(sample_num)]


# -------------------- Build Groups --------------------
def stack_indices(idxs): return np.vstack([features_per_sample[i] for i in idxs])

X_train_PB = stack_indices(train_pristine_indices)
X_valid_PB = stack_indices(valid_pristine_indices)
X_PB_0     = features_per_sample[test_pristine_indices[0]]
X_KB       = [features_per_sample[i] for i in range(10)]


# -------------------- Filtering: keep only 95% majority --------------------
# keep_idx, out_idx = filter_by_iforest(X_train_PB, keep_ratio=keep_ratio)
# X_train_PB_filtered = X_train_PB[keep_idx]
# X_train_PB_outliers = X_train_PB[out_idx]
# print(f"Filtered {len(out_idx)} outliers out of {X_train_PB.shape[0]} training samples.")


# -------------------- Prepare Data for t-SNE --------------------
# groups = [X_train_PB_filtered, X_valid_PB, X_PB_0] + X_KB
groups = [X_train_PB, X_valid_PB, X_PB_0] + X_KB
X = np.vstack(groups)
N = X.shape[0]
perpl = min(30, max(5, (N - 1)//3))

tsne = TSNE(n_components=2, random_state=42, perplexity=perpl, init='pca', learning_rate='auto')
X2d = tsne.fit_transform(X)


# -------------------- Group Indexing --------------------
sizes  = [g.shape[0] for g in groups]
starts = np.cumsum([0] + sizes[:-1])
ends   = np.cumsum(sizes)
slices = [slice(s, e) for s, e in zip(starts, ends)]


# -------------------- KDE Contour for Train PB --------------------
xy = X2d[slices[0]].T
kde = gaussian_kde(xy)
pad = 0.1
x_min, x_max = xy[0].min(), xy[0].max()
y_min, y_max = xy[1].min(), xy[1].max()
dx, dy = (x_max - x_min) * pad, (y_max - y_min) * pad
Xg, Yg = np.mgrid[x_min-dx:x_max+dx:200j, y_min-dy:y_max+dy:200j]
Z = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
flat = Z.ravel()
order = np.argsort(flat)[::-1]
cdf = np.cumsum(flat[order])
level = flat[order][np.searchsorted(cdf, 0.9 * cdf[-1])]


# -------------------- Plot --------------------
plt.figure(figsize=(12, 9))

# (1) Filtered Training PB
plt.scatter(*X2d[slices[0]].T, c='C0', marker='^', s=15, alpha=0.6, label=f'Training Pristine (n={sizes[0]})')


# (2) Filtered-out outliers (projected via same t-SNE)
# Need to transform them with the same t-SNE embedding
# X2d_outliers = tsne.fit_transform(np.vstack([X_train_PB_filtered, X_train_PB_outliers]))
# X2d_out = X2d_outliers[-len(X_train_PB_outliers):]
# plt.scatter(X2d_out[:, 0], X2d_out[:, 1], c='red', marker='x', s=25, label='Filtered-out (Outliers)')

# Contour of density
cs = plt.contour(Xg, Yg, Z, levels=[level], linestyles='--', colors='k', linewidths=1)
# plt.clabel(cs, fmt={level: '90% Density Contour'}, inline=True)

# (3) Valid/Test PB
v_idx = [i - 9 for i in valid_pristine_indices]
plt.scatter(*X2d[slices[1]].T, c='C1', marker='^', s=15, alpha=0.6,
            label=f'Thresholding Pristine #{v_idx[0]} & #{v_idx[1]} (n={sizes[1]})')
t_idx = test_pristine_indices[0] - 9
plt.scatter(*X2d[slices[2]].T, c='C2', marker='^', s=15, alpha=0.6,
            label=f'Testing Pristine #{t_idx} (n={sizes[2]})')

# (4) KB samples
tab10 = plt.cm.get_cmap('tab10', 10)
for i in range(10):
    sl = slices[3 + i]
    plt.scatter(*X2d[sl].T, c=[tab10(i)], edgecolor='black', s=40, label=f'Weak-Adhesion #{11 + i} (n={sizes[3+i]})')

plt.xlabel('t-SNE dim 1', fontsize=14)
plt.ylabel('t-SNE dim 2', fontsize=14)
plt.xlim((-120, 120))
plt.ylim((-120, 120))
plt.title('t-SNE of PB & KB Features')
plt.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, -0.12), # Centers it below the x-axis
    ncol=4,                      # Adjust columns based on visual preference
    fontsize=12,
    frameon=True
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22) # Reserve space for the legend


# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
# plt.tight_layout()
plt.savefig(os.path.join(file_path, model_selection+'_t-SNE_'+feat_selection+'.png'), bbox_inches='tight')
plt.show()


