from utlis import *
from models import *
from torch.utils.data import TensorDataset, DataLoader
import time


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')
# torch.backends.cudnn.benchmark = True


# ----------------------------
# Hyperparameters
# ----------------------------
model_selection = 'EngAD'  # EngWAD | EngAD | WAD | AD
date = '0710_A_3_8_10'
OOD_sample_1 = 14
OOD_sample_2 = 16
depth = 274

batch_size = 256
epochs = 1000
epsilon = 1e-4

r = 4
h_dim = 128
p_dim = 128
wasser_pattern = 'avg'  # 'avg' or 'min'

# loss weights
if model_selection == 'EngWAD':
    lambda_consis = 1.0
    lambda_recons = 2.0
    lambda_wasser = 10.0
elif model_selection == 'EngAD':
    lambda_consis = 1.0
    lambda_recons = 2.0
    lambda_wasser = 0.0
elif model_selection == 'WAD':
    lambda_consis = 0.0
    lambda_recons = 1.0
    lambda_wasser = 5.0
elif model_selection == 'AD':
    lambda_consis = 0.0
    lambda_recons = 1.0
    lambda_wasser = 0.0

# Sinkhorn OT params
sink_eps   = 0.02
sink_iters = 50
softmin_T  = 50.0

# cosine + warmup scheduler hyperparams
base_lr       = 1e-3
base_wd       = 1e-4
lr_min        = 1e-6
wd_min        = 1e-5
warmup_epochs = 10   # set 0 to disable warmup


# ----------------------------
# Data pre-processing
# ----------------------------
current_path = os.getcwd()
file_path = os.path.join(current_path, 'data', 'ascan_all_0710_A.csv')
X_all = np.genfromtxt(file_path, delimiter=',').astype('f4')
X_all = torch.from_numpy(X_all).to(device).unsqueeze(1)  # [N, 1, depth_full]
X_all = X_all[:, :, :depth]                              # ensure length=depth

# 0710_A geometry: create piece labels 0..19
nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36]
widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29]
nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]
counts  = [(nbeam_U[k] - nbeam_L[k]) * (nscan_U[k] - nscan_L[k]) for k in range(len(nbeam_L))]
edges   = np.cumsum([0] + counts)
N       = edges[-1]

labels = np.zeros(N, dtype=np.int64)
for k in range(20):
    labels[edges[k]:edges[k+1]] = k
labels = torch.from_numpy(labels).float().view(N, 1, 1).to(device)
X_all_labeled = torch.cat([X_all, labels], dim=2)  # [N, 1, depth+1] (depth+1 = 275)

# training split logic based on 'date'
parts = date.split('_')
nums = [int(x) for x in parts[2:]]          # three indices
nums_sorted = sorted(nums[:2])
train_PB_indices = list(range(10, 20))      # PB: 10..19
for idx in nums:
    train_PB_indices.remove(idx + 9)

if OOD_sample_1 and OOD_sample_2:
    train_aug_KB_index = [OOD_sample_1-11, OOD_sample_2-11]
else:
    train_aug_KB_index = random.sample(range(10), 2)  # pick 2 KB indices for OOD Wasserstein loss
train_aug_KB_index_sorted = sorted([i+11 for i in train_aug_KB_index])
print('Wasserstein Augmented Samples: ', train_aug_KB_index_sorted)

train_mask = torch.isin(labels.squeeze(), torch.tensor(train_PB_indices, device=device))
train_aug_mask = torch.isin(labels.squeeze(), torch.tensor(train_aug_KB_index, device=device))
X_train = X_all_labeled[train_mask]          # [N_train, 1, 275]
X_train_aug = X_all_labeled[train_aug_mask]  # [N_aug,   1, 275]
perm = torch.randperm(X_train.size(0), device=device)
X_train = X_train[perm]

X_feat = per_series_zscore(X_train[:, :, :depth]).float()
y_label = X_train[:, :, depth].long().squeeze(-1).squeeze(-1)
X_train = torch.cat([X_feat, y_label.view(-1,1,1).float()], dim=2)
ds = TensorDataset(X_train[:, :, :depth], y_label)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

X_aug_feat  = per_series_zscore(X_train_aug[:, :, :depth]).float()
y_aug_label = X_train_aug[:, :, depth].long().squeeze(-1).squeeze(-1)
ds_aug = TensorDataset(X_aug_feat, y_aug_label)
dl_aug = DataLoader(ds_aug, batch_size=batch_size, shuffle=True, drop_last=False)
aug_iter = iter(dl_aug)


# ----------------------------
# Model
# ----------------------------
# model = ContraAE(c_in=1, r=r, h_dim=h_dim, p_dim=p_dim).to(device)
model = TransAE(c_in=1, r=r, h_dim=h_dim, p_dim=p_dim).to(device)

# split params for weight decay
decay, no_decay = [], []
for name, p in model.named_parameters():
    if not p.requires_grad: continue
    if ('bias' in name) or ('bn' in name) or ('norm' in name) or ('running_' in name):
        no_decay.append(p)
    else:
        decay.append(p)
opt = torch.optim.AdamW([
        {'params': decay,    'weight_decay': base_wd},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=base_lr)
augment = aug_1d_univariate()


# ----------------------------
# Logging containers
# ----------------------------
loss_history   = []
consis_history = []
recons_history = []
wasser_history = []
contra_history = []  # if you later add VICReg loss here


# ----------------------------
# Training
# ----------------------------
model.train()
start = time.perf_counter()
for ep in range(1, epochs+1):
    # --- epoch-wise LR/WD schedule (cosine + warmup) ---
    if ep <= warmup_epochs:
        lr_now = base_lr * warmup_factor(ep-1, warmup_epochs)
        wd_now = base_wd * warmup_factor(ep-1, warmup_epochs)
    else:
        t = ep - warmup_epochs - 1
        T = max(1, epochs - warmup_epochs)
        lr_now = cosine_schedule(base_lr, lr_min, t, T)
        wd_now = cosine_schedule(base_wd, wd_min, t, T)

    for pg in opt.param_groups:
        pg['lr'] = lr_now
        if pg.get('weight_decay', None) is not None and pg['weight_decay'] > 0.0:
            pg['weight_decay'] = wd_now

    running = {"loss":0.0, "consis":0.0, "recons":0.0, "wasser":0.0, "contra":0.0}
    nsteps = 0

    for xb, yb in dl:   # xb: (B, 1, depth) pristine signals, yb: (B,)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # (A) Clean path: consistency + reconstruction on z
        # expect encode_to_proj(x, return_fw=True, decode=True) -> z, h, p, z_fw, out
        z, h, _, z_fw, out = model.encode_to_proj(xb, return_fw=True, decode=True)
        loss_recons  = F.mse_loss(out, xb)
        loss_consis  = consis_var_loss(z_fw.float(), reduction="mean")

        # (B) VICReg on projected features p using two augmented views (optional)
        # v1 = torch.stack([augment(x) for x in xb], 0)
        # v2 = torch.stack([augment(x) for x in xb], 0)
        # _, _, p1 = model.encode_to_proj(v1)    # normalized proj
        # _, _, p2 = model.encode_to_proj(v2)
        # loss_vic, _, _, _ = vicreg_loss(p1.float(), p2.float(), sim_w=25.0, var_w=15.0, cov_w=10.0, gamma=1.0)
        loss_vic = 0.0

        # (C) Wasserstein min-pair on h: maximize min_{j,k} W2(Vk, Uj)
        # try:
        #     xq, yq = next(aug_iter)  # OOD batch with labels
        # except StopIteration:
        #     aug_iter = iter(dl_aug)
        #     xq, yq = next(aug_iter)

        xq, yq = X_aug_feat, y_aug_label

        xq = xq.to(device, non_blocking=True)
        yq = yq.to(device, non_blocking=True)

        _, h_ood, _ = model.encode_to_proj(xq)  # (Bq, h_dim), normalized by head

        uniq_pb  = torch.unique(yb)             # pristine labels in this step
        uniq_ood = torch.unique(yq)             # OOD labels in this step

        pair_w = []
        if wasser_pattern == 'min':
            for j in uniq_ood:
                Uj = h_ood[yq == j]
                if Uj.numel() == 0: continue
                for k in uniq_pb:
                    Vk = h[yb == k]
                    if Vk.numel() == 0: continue
                    Wjk = sinkhorn_w2_torch(Vk, Uj, eps=sink_eps, n_iters=sink_iters)
                    pair_w.append(Wjk)

            if pair_w:
                W_all = torch.stack(pair_w)
                wasserstein_dist = softmin(W_all, alpha=softmin_T)
            else:
                wasserstein_dist = torch.tensor(0.0, device=device)
        elif wasser_pattern == 'avg':
            W_min = []
            for j in uniq_ood:
                Uj = h_ood[yq == j]
                if Uj.numel() == 0: continue
                for k in uniq_pb:
                    Vk = h[yb == k]
                    if Vk.numel() == 0: continue
                    Wjk = sinkhorn_w2_torch(Vk, Uj, eps=sink_eps, n_iters=sink_iters)
                    pair_w.append(Wjk)
                W_min.append(softmin(torch.stack(pair_w), alpha=softmin_T))
                pair_w = []
            
            if W_min:
                W_all = torch.stack(W_min)
                wasserstein_dist = torch.mean(W_all)
            else:
                wasserstein_dist = torch.tensor(0.0, device=device)
            

        # Total loss
        loss = (loss_vic
                + lambda_consis * loss_consis
                + lambda_recons * loss_recons
                - lambda_wasser * wasserstein_dist)

        # backprop
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        # logging
        running["loss"]   += float(loss)
        running["consis"] += float(loss_consis)
        running["recons"] += float(loss_recons)
        running["wasser"] += float(wasserstein_dist)
        running["contra"] += float(loss_vic)
        nsteps += 1

    # epoch averages
    avg = {k: running[k]/max(1, nsteps) for k in running}
    print(f"epoch {ep:04d} | lr {lr_now:.3e} wd {wd_now:.3e} | loss {avg['loss']:.6f} "
          f"(consis {avg['consis']:.6f} + recons {avg['recons']:.6f} + wasser {avg['wasser']:.6f} + contra {avg['contra']:.6f})")

    # histories
    loss_history.append(avg["loss"])
    recons_history.append(avg["recons"])
    consis_history.append(avg["consis"])
    wasser_history.append(avg["wasser"])
    # contra_history.append(avg["contra"])

    if ep >= 2:
        delta_recons = np.abs(recons_history[ep-1]-recons_history[ep-2]) / np.abs(recons_history[ep-2])
        delta_consis = np.abs(consis_history[ep-1]-consis_history[ep-2]) / np.abs(consis_history[ep-2])
        delta_wasser = np.abs(wasser_history[ep-1]-wasser_history[ep-2]) / np.abs(wasser_history[ep-2])
        if delta_recons <= epsilon and delta_consis <= epsilon and delta_wasser <= epsilon:
            break

end = time.perf_counter()
elapsed = end - start
print(f"Elapsed time: {elapsed:.6f} seconds")

os.makedirs('model_para', exist_ok=True)
torch.save(model.state_dict(), f'model_para/{wasser_pattern}/{model_selection}/{model_selection}_{wasser_pattern}_{date}_OOD_{train_aug_KB_index_sorted[0]}_{train_aug_KB_index_sorted[1]}.pth')


# ----------------------------
# Plot losses (log-scale)
# ----------------------------
epochs_range = np.arange(1, len(loss_history)+1)
plt.figure(figsize=(9, 6))
plt.plot(epochs_range, loss_history,   label='Total Loss')
plt.plot(epochs_range, consis_history, label='Consistency')
plt.plot(epochs_range, recons_history, label='Reconstruction')
plt.plot(epochs_range, wasser_history, label='Wasserstein')
if any(c != 0.0 for c in contra_history):
    plt.plot(epochs_range, contra_history, label='VICReg')

plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log scale)')
plt.title(f'{model_selection} Training Loss — {date}')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.4)
plt.tight_layout()
plt.savefig(f'{model_selection}_loss_curve_{wasser_pattern}_{date}_OOD_{train_aug_KB_index_sorted[0]}_{train_aug_KB_index_sorted[1]}.png', dpi=300)
plt.show()

