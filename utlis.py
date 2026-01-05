from __future__ import print_function
import numpy as np
import os
import os.path
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.image as mpimg
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data
from torchvision import transforms
from typing import Callable
from timeit import default_timer as timer
import math
import abc
from typing import Union
import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
import ot
import random


device = ('cuda' if torch.cuda.is_available() else 'cpu')


def read_file(file_path, number_array, number_array_batch):
    data = np.genfromtxt(file_path, delimiter=',')
    data = data[1:, 2:]
    number_beam = number_array - number_array_batch + 1
    scanning_length = int(data.size/(number_beam*data.shape[1]))
    
    cscan_data = np.zeros((number_beam, scanning_length, data.shape[1]))
    for j in range(scanning_length):
        cscan_data[:, j, :] = data[j*number_beam:(j+1)*number_beam, :]

    return cscan_data


def plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    fig.subplots_adjust(right = 1)
    colors = plt.cm.plasma(data)
    ax.voxels(data, facecolors = colors)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    m = cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    m.set_array([])
    ax.set_xlabel("Phase Array Beam ID")
    ax.set_ylabel("Scanning Legnth")
    ax.set_zlabel("Thickness")
    ax.invert_zaxis()
    plt.show()


def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input-lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


def feature_map_permute(input):
    s = input.data.shape
    l = len(s)

    # permute feature channel to the last:
    # NxCxL --> NxLxC
    if l == 2:
        x = input # NxC
    elif l == 3:
        x = input.permute(0, 2, 1)
    elif l == 4:
        x = input.permute(0, 2, 3, 1)
    elif l == 5:
        x = input.permute(0, 2, 3, 4, 1)
    else:
        x = []
        print('wrong feature map size')
    x = x.contiguous()
    # NxLxC --> (NxL)xC
    x = x.view(-1, s[1])
    return x


class EntropyLoss(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        b = x * torch.log(x + self.eps)
        b = -1.0 * b.sum(dim=1)
        b = b.mean()
        return b
    

class EntropyLossEncap(nn.Module):
    def __init__(self, eps = 1e-12):
        super(EntropyLossEncap, self).__init__()
        self.eps = eps
        self.entropy_loss = EntropyLoss(eps)

    def forward(self, input):
        score = feature_map_permute(input)
        ent_loss_val = self.entropy_loss(score)
        return ent_loss_val
    

class MeanAbsoluteRelativeError(nn.Module):
    def __init__(self, eps = 1e-12):
        super(MeanAbsoluteRelativeError, self).__init__()
        self.eps = eps

    def forward(self, pred, true):
        mare = torch.div(torch.abs(true - pred), torch.abs(true + self.eps))
        mare = mare.sum(dim=-1)
        mare = mare.mean()
        return mare
    

class SoftThresholding(nn.Module):
    def __init__(self):
        super(SoftThresholding, self).__init__()

    def forward(self, input, lambd):
        output = torch.sign(input) * nn.functional.relu(torch.abs(input) - lambd)
        return output


class L21Norm(nn.Module):
    def __init__(self):
        super(L21Norm, self).__init__()

    def forward(self, pred, truth):
        input = pred - truth
        input = torch.transpose(input.squeeze(), 0, 1)
        output = torch.mean(torch.norm(input, dim=0))
        return output
    

class RelativeL21Norm(nn.Module):
    def __init__(self, eps=1e-6):
        super(RelativeL21Norm, self).__init__()
        self.eps = eps

    def forward(self, pred, truth):
        input = (pred - truth) / (truth + self.eps)
        input = torch.transpose(input.squeeze(), 0, 1)
        output = torch.mean(torch.norm(input, dim=0))
        return output


class MVTEC(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 category='carpet', resize=None, interpolation=2, train_defect=False):
        # 0: InterpolationMode.NEAREST,  1: InterpolationMode.LANCZOS
        # 2: InterpolationMode.BILINEAR, 3: InterpolationMode.BICUBIC
        # 4: InterpolationMode.BOX,      5: InterpolationMode.HAMMING
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = resize
        self.interpolation = interpolation
        self.train_defect = train_defect
        
        # load images for training
        if self.train:
            self.train_data = []
            self.train_labels = []
            self.train_indices = []
            cwd = os.getcwd()
            if self.train_defect:
                trainFolder = self.root + '/' + category + '/train/defect/'
            else:
                trainFolder = self.root + '/' + category + '/train/good/'
            os.chdir(trainFolder)
            filenames = [f.name for f in os.scandir()]
            for file in filenames:
                img = mpimg.imread(file)
                if img.shape[2] == 4:
                    img = img[:, :, 0:3]
                if file[-3:] != 'bmp':
                    img = img * 255
                img = img.astype(np.uint8)
                # plt.imshow(img)
                # plt.show()
                self.train_data.append(img)
                self.train_labels.append(1)
                self.train_indices.append(int(file[:-4]))   
            os.chdir(cwd)
                
            self.train_data = np.array(self.train_data)      
        else:
        # load images for testing
            self.test_data = []
            self.test_labels = []
            self.test_indices = []
            
            cwd = os.getcwd()
            testFolder = self.root + '/' + category + '/test/'
            os.chdir(testFolder)
            subfolders = [sf.name for sf in os.scandir() if sf.is_dir()]
            cwsd = os.getcwd()
            
            # for every subfolder in test folder
            for subfolder in subfolders:
                label = 0
                if subfolder == 'good':
                    label = 1
                testSubfolder = './' + subfolder + '/'
                os.chdir(testSubfolder)
                filenames = [f.name for f in os.scandir()]
                for file in filenames:
                    img = mpimg.imread(file)
                    if img.shape[2] == 4:
                        img = img[:, :, 0:3]
                    if file[-3:] != 'bmp':
                        img = img * 255
                    img = img.astype(np.uint8)
                    self.test_data.append(img)
                    self.test_labels.append(label)
                    self.test_indices.append(int(file[:-4]))
                os.chdir(cwsd)
            os.chdir(cwd)
                
            self.test_data = np.array(self.test_data)
                
    def __getitem__(self, index):
        if self.train:
            img, target, idx = self.train_data[index], self.train_labels[index], self.train_indices[index]
        else:
            img, target, idx = self.test_data[index], self.test_labels[index], self.test_indices[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        # resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize, self.interpolation)
            img = resizeTransf(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, idx

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    

def _get_corner_min_array(f_mat: np.ndarray, i: int, j: int) -> float:
    if i > 0 and j > 0:
        a = min(f_mat[i - 1, j - 1],
                f_mat[i, j - 1],
                f_mat[i - 1, j])
    elif i == 0 and j == 0:
        a = f_mat[i, j]
    elif i == 0:
        a = f_mat[i, j - 1]
    else:  # j == 0:
        a = f_mat[i - 1, j]
    return a


def _bresenham_pairs(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Generates the diagonal coordinates

    Parameters
    ----------
    x0 : int
        Origin x value
    y0 : int
        Origin y value
    x1 : int
        Target x value
    y1 : int
        Target y value

    Returns
    -------
    np.ndarray
        Array with the diagonal coordinates
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dim = max(dx, dy)
    pairs = np.zeros((dim, 2), dtype=np.int64)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx // 2
        for i in range(dx):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy // 2
        for i in range(dy):
            pairs[i, 0] = x
            pairs[i, 1] = y
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return pairs


def _fast_distance_matrix(p, q, diag, dist_func):
    n_diag = diag.shape[0]
    diag_max = 0.0
    i_min = 0
    j_min = 0
    p_count = p.shape[0]
    q_count = q.shape[0]

    # Create the distance array
    dist = np.full((p_count, q_count), np.inf, dtype=np.float64)

    # Fill in the diagonal with the seed distance values
    for k in range(n_diag):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        d = dist_func(p[i0], q[j0])
        diag_max = max(diag_max, d)
        dist[i0, j0] = d

    for k in range(n_diag - 1):
        i0 = diag[k, 0]
        j0 = diag[k, 1]
        p_i0 = p[i0]
        q_j0 = q[j0]

        for i in range(i0 + 1, p_count):
            if np.isinf(dist[i, j0]):
                d = dist_func(p[i], q_j0)
                if d < diag_max or i < i_min:
                    dist[i, j0] = d
                else:
                    break
            else:
                break
        i_min = i

        for j in range(j0 + 1, q_count):
            if np.isinf(dist[i0, j]):
                d = dist_func(p_i0, q[j])
                if d < diag_max or j < j_min:
                    dist[i0, j] = d
                else:
                    break
            else:
                break
        j_min = j
    return dist


def _fast_frechet_matrix(dist: np.ndarray, diag: np.ndarray,
                         p: np.ndarray, q: np.ndarray) -> np.ndarray:

    for k in range(diag.shape[0]):
        i0 = diag[k, 0]
        j0 = diag[k, 1]

        for i in range(i0, p.shape[0]):
            if np.isfinite(dist[i, j0]):
                c = _get_corner_min_array(dist, i, j0)
                if c > dist[i, j0]:
                    dist[i, j0] = c
            else:
                break

        # Add 1 to j0 to avoid recalculating the diagonal
        for j in range(j0 + 1, q.shape[0]):
            if np.isfinite(dist[i0, j]):
                c = _get_corner_min_array(dist, i0, j)
                if c > dist[i0, j]:
                    dist[i0, j] = c
            else:
                break
    return dist


def _fdfd_matrix(p: np.ndarray, q: np.ndarray,
                 dist_func: Callable[[np.array, np.array], float]) -> float:
    diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
    ca = _fast_distance_matrix(p, q, diagonal, dist_func)
    ca = _fast_frechet_matrix(ca, diagonal, p, q)
    return ca


class FastDiscreteFrechetMatrix(object):
    def __init__(self, dist_func):
        self.times = []
        self.dist_func = dist_func
        self.ca = np.zeros((1, 1))
        # JIT the numba code
        self.distance(np.array([[0.0, 0.0], [1.0, 1.0]]),
                      np.array([[0.0, 0.0], [1.0, 1.0]]))

    def timed_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        start = timer()
        diagonal = _bresenham_pairs(0, 0, p.shape[0], q.shape[0])
        self.times.append(timer() - start)

        start = timer()
        ca = _fast_distance_matrix(p, q, diagonal, self.dist_func)
        self.times.append(timer() - start)

        start = timer()
        ca = _fast_frechet_matrix(ca, diagonal, p, q)
        self.times.append(timer() - start)

        self.ca = ca
        return ca[p.shape[0]-1, q.shape[0]-1]

    def distance(self, p: np.ndarray, q: np.ndarray) -> float:
        ca = _fdfd_matrix(p, q, self.dist_func)
        self.ca = ca
        return ca[p.shape[0]-1, q.shape[0]-1]


def haversine(p: np.ndarray, q: np.ndarray) -> float:
    d = q - p
    a = math.sin(d[0]/2.0)**2 + math.cos(p[0]) * math.cos(q[0]) \
        * math.sin(d[1]/2.0)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return c


def earth_haversine(p: np.ndarray, q: np.ndarray) -> float:
    earth_radius = 6378137.0
    return haversine(np.radians(p), np.radians(q)) * earth_radius


def euclidean(p: np.ndarray, q: np.ndarray) -> float:
    d = p - q
    return math.sqrt(np.dot(d, d))


def memory_item_aggregation(memory_bank, l=2):
    r = int(np.ceil(l / 2))
    s = memory_bank.shape
    memory_bank_agg = torch.ones(s[0], s[1], int(s[-1]*(2*r+1))).to(device)
    for i, memory_instance in enumerate(memory_bank):
        for j, _ in enumerate(memory_instance):
            if j < r:
                memory_bank_agg[i, j, :] = torch.cat((memory_instance[0, :].tile((r-j, 1)), memory_instance[0:j+r+1, :]), 0).view(-1)
            elif j >= s[1]-r:
                memory_bank_agg[i, j, :] = torch.cat((memory_instance[j-r:, :], memory_instance[-1, :].tile((j-s[1]+r+1, 1))), 0).view(-1)
            else:
                memory_bank_agg[i, j, :] = memory_instance[j-r:j+r+1, :].view(-1)

    memory_bank_agg = memory_bank_agg.view(-1, memory_bank_agg.shape[-1])
    # memory_bank_agg = memory_bank_agg[~torch.all(memory_bank_agg == torch.zeros(c), dim=1)]
    return memory_bank_agg

'''
def MemoryBankSearch(model, decoder, training, testing, dist=euclidean, r=20):
    _, _, hidden_features_training = model(training)
    _, _, hidden_features_testing = model(testing)

    hidden_features_training = hidden_features_training.permute(0, 2, 1)
    hidden_features_testing = hidden_features_testing.permute(0, 2, 1)
    hidden_features_training_raw = hidden_features_training.view(-1, hidden_features_training.shape[-1])
    hidden_features_training_agg = MemoryItemAggregation(hidden_features_training)

    fdfdm = FastDiscreteFrechetMatrix(dist)
    t = np.linspace(0, testing.shape[2]-1, num=testing.shape[2])
    anomaly_score_hidden = np.zeros(testing.shape[0])
    anomaly_score_recon = np.zeros(testing.shape[0])
    for i, hidden_feature_testing in enumerate(hidden_features_testing):
        hidden_feature_testing_agg = MemoryItemAggregation(hidden_feature_testing.unsqueeze(0))
        feature_distances = np.zeros(hidden_feature_testing_agg.shape[0])
        hidden_feature_memory = torch.zeros(hidden_feature_testing.shape).to(device)
        for j, hidden_rep_testing in enumerate(hidden_feature_testing_agg):
            rep_distances = np.zeros(hidden_features_training_agg.shape[0])
            for k, hidden_rep_training in enumerate(hidden_features_training_agg):
                rep_distances[k] = 1 - F.cosine_similarity(hidden_rep_testing.view(1, -1), hidden_rep_training.view(1, -1)).item()

            min_indices = np.argpartition(rep_distances, r)
            feature_distances[j] = np.max([np.mean(rep_distances[min_indices[:r]]), 0])  # Average over r smallest distances
            hidden_feature_memory[j, :] = torch.mean(hidden_features_training_raw[min_indices[:r], :], dim=0)

        # Distance in hidden        
        anomaly_score_hidden[i] = np.mean(feature_distances)

        # Frechet distance in recon
        hidden_feature_memory = hidden_feature_memory.permute(1, 0).unsqueeze(0)
        recon_memory = decoder(hidden_feature_memory)
        ascan_recon_memory = np.stack((t, np.squeeze(recon_memory.detach().cpu().numpy())), axis=-1)
        ascan_testing = np.stack((t, np.squeeze(testing[i, :].detach().cpu().numpy())), axis=-1)   
        anomaly_score_recon[i] = fdfdm.distance(ascan_recon_memory, ascan_testing)

    return anomaly_score_hidden, anomaly_score_recon
'''

def memory_processing(model, decoder, features_training, testing, interface_start, interface_end, r, R, dist=euclidean):
    _, _, _, hidden_features_testing = model(testing)
    fdfdm = FastDiscreteFrechetMatrix(dist)
    # t = np.linspace(0, testing.shape[2]-1, num=testing.shape[2])
    t = np.linspace(0, interface_end-interface_start-1, num=interface_end-interface_start)
    anomaly_score_hidden = np.zeros(testing.shape[0])
    anomaly_score_recon_Frechet = np.zeros(testing.shape[0])
    anomaly_score_recon_Euclidean = np.zeros(testing.shape[0])

    for i, hidden_feature_testing in enumerate(hidden_features_testing):
        feature_distances = np.zeros(features_training.shape[0])
        for j, hidden_feature_training in enumerate(features_training):
            hidden_feature_testing = hidden_feature_testing.transpose(0, 1).reshape(1, -1)
            hidden_feature_training = hidden_feature_training.transpose(0, 1).reshape(1, -1)
            feature_distances[j] = 1 - F.cosine_similarity(hidden_feature_testing, hidden_feature_training).item()
            # diff = hidden_feature_testing[:, r:] - hidden_feature_training[:, r:]
            # feature_distances[j] = torch.norm(diff, p=2, dim=0).sum().item()
            
        # Distance in hidden space
        min_indices = np.argpartition(feature_distances, R)
        anomaly_score_hidden_value = np.median(feature_distances[min_indices[:R]])
        anomaly_score_hidden[i] = np.max([anomaly_score_hidden_value, 0])

        # Frechet & Euclidean distance
        anomaly_score_hidden_value_index = np.where(feature_distances==anomaly_score_hidden_value)[0][0]
        hidden_feature_memory = features_training[anomaly_score_hidden_value_index].unsqueeze(0)
        recon_memory = decoder(hidden_feature_memory)
        ascan_recon_memory = np.stack((t, np.squeeze(recon_memory[:, :, interface_start:interface_end].detach().cpu().numpy())), axis=-1)
        ascan_testing = np.stack((t, np.squeeze(testing[i, :, interface_start:interface_end].detach().cpu().numpy())), axis=-1)   
        anomaly_score_recon_Frechet[i] = fdfdm.distance(ascan_recon_memory, ascan_testing)
        anomaly_score_recon_Euclidean[i] = np.linalg.norm(ascan_recon_memory-ascan_testing, 'fro')

    return anomaly_score_hidden, anomaly_score_recon_Frechet, anomaly_score_recon_Euclidean


def memory_processing_fast_AE(model,
                              decoder,
                              features_training,    # tensor, shape [N_train, C, H]
                              testing,              # tensor, shape [N_test, C, T]
                              interface_start: int,
                              interface_end: int,
                              r, R: int,
                              dist=euclidean):
    # 1) get hidden features for all test samples at once
    #    assume model(testing) returns (..., hidden_features_testing)
    # hidden_features_testing = model(testing)[3]  # [N_test, C, H]
    hidden_features_testing = model(testing)
    
    # 2) move to CPU+numpy and flatten
    hidden_np = hidden_features_testing.detach().cpu().numpy()
    train_np  = features_training.detach().cpu().numpy()
    N_test, C, H = hidden_np.shape
    N_train      = train_np.shape[0]
    
    # flatten each sample to 1‑D
    hidden_flat = hidden_np.reshape(N_test, -1)     # [N_test, C*H]
    train_flat  = train_np .reshape(N_train,  -1)   # [N_train, C*H]
    
    # 3) cosine‑distance matrix in one shot
    #    sim[i,j] = cosine_similarity(hidden_flat[i], train_flat[j])
    dot = hidden_flat @ train_flat.T                             # [N_test, N_train]
    hn  = np.linalg.norm(hidden_flat, axis=1, keepdims=True)     # [N_test, 1]
    tn  = np.linalg.norm(train_flat,  axis=1, keepdims=True).T  # [1, N_train]
    sim = dot / (hn @ tn)                                        # broadcast → [N_test, N_train]
    dist_mat = 1.0 - sim                                         # cosine distance
    
    # 4) for each test sample, find the R smallest distances and their median
    idx_R    = np.argpartition(dist_mat, R, axis=1)[:, :R]   # [N_test, R]
    dist_R   = np.take_along_axis(dist_mat, idx_R, axis=1)   # [N_test, R]
    medians  = np.median(dist_R, axis=1)                     # [N_test]
    anomaly_score_hidden = np.clip(medians, 0, None)         # ensure ≥0
    
    # 5) pick, for each test sample, the training‐index whose dist is “closest” to that median
    #    (we avoid direct float == comparisons)
    offsets = np.abs(dist_R - medians[:, None])              # [N_test, R]
    choose  = np.argmin(offsets, axis=1)                     # [N_test]
    mem_idx = idx_R[np.arange(N_test), choose]               # [N_test]
    
    # 6) precompute your time‐axis & test‐window once
    L       = interface_end - interface_start
    t       = np.linspace(0, L-1, num=L)
    test_seg = testing[:, 0, interface_start:interface_end].detach().cpu().numpy()                 # [N_test, L]
    
    # 7) batch‐decode all the chosen “memories”
    #    bring mem_idx back to a torch tensor on the right device:
    dev         = features_training.device
    mem_idx_t   = torch.as_tensor(mem_idx, dtype=torch.long, device=dev)
    hidden_mem  = features_training[mem_idx_t]               # [N_test, C, H]
    recon_all   = decoder(hidden_mem)                        # [N_test, C_out, T_out]
    recon_seg   = recon_all[:, 0, interface_start:interface_end].detach().cpu().numpy()                 # [N_test, L]
    
    # 8) now only a small Python loop remains for Frechet & Frobenius
    fdfdm = FastDiscreteFrechetMatrix(dist)
    anomaly_score_recon_Frechet   = np.zeros(N_test)
    anomaly_score_recon_Euclidean = np.zeros(N_test)
    
    for i in range(N_test):
        ascan_mem  = np.column_stack((t, recon_seg[i]))
        ascan_test = np.column_stack((t, test_seg[i]))
        anomaly_score_recon_Frechet[i]   = fdfdm.distance(ascan_mem, ascan_test)
        anomaly_score_recon_Euclidean[i] = np.linalg.norm(ascan_mem - ascan_test, 'fro')

    return anomaly_score_hidden, anomaly_score_recon_Frechet, anomaly_score_recon_Euclidean


@torch.no_grad()
def memory_processing_fast_SimCLR(model,
                                  features_training: torch.Tensor,   # [N_train, H]
                                  testing: torch.Tensor,             # [N_test, C, T]
                                  R: int):
    """
    Returns:
      anomaly_score_embedding: (N_test,)  Euclidean distance between each test embedding
                                     and its selected 'memory' embedding (closest-to-median among R-NN).
    """
    device = features_training.device
    model = model.to(device).eval()
    testing = testing.to(device)

    # 1) Test embeddings
    z_test = model(testing)                       # [N_test, H]
    # Ensure L2-normalized (SimCLR encoder already does this, but safe to normalize)
    z_test = F.normalize(z_test, dim=1)

    # 2) Training embeddings (ensure normalized)
    Z_train = F.normalize(features_training.to(device), dim=1)  # [N_train, H]

    # 3) Cosine distance matrix: for normalized vectors, sim = z_test @ Z_train.T
    #    dist = 1 - sim
    sim = z_test @ Z_train.T                       # [N_test, N_train]
    dist_mat = 1.0 - sim                           # cosine distance

    # 4) For each test, take the R smallest distances
    #    Use topk with largest=False for efficiency (no full sort)
    dist_R, idx_R = torch.topk(dist_mat, k=R, dim=1, largest=False)  # both [N_test, R]

    # 5) Median distance (robust center) and pick neighbor closest to that median
    #    For even R, torch.median returns the lower median; that's fine/robust.
    medians = dist_R.median(dim=1).values          # [N_test]
    offsets = (dist_R - medians[:, None]).abs()    # [N_test, R]
    choose = offsets.argmin(dim=1)                 # [N_test]
    mem_idx = idx_R[torch.arange(idx_R.size(0), device=device), choose]  # [N_test]

    # 6) Pull memory embeddings and compute Euclidean distance in embedding space
    mem_emb = Z_train[mem_idx]                     # [N_test, H]
    # Euclidean distance between embeddings (vectorized)
    anomaly_score_embedding = torch.linalg.norm(z_test - mem_emb, dim=1)  # [N_test]

    return anomaly_score_embedding.detach().cpu().numpy()


@torch.no_grad()
def memory_processing_fast_ContraAE(model,
                                    features_training: torch.Tensor,   # [N_train, C1, H]
                                    testing: torch.Tensor,             # [N_test, C2, T]
                                    R: int,
                                    feat_selection: str):
    """
    Compute memory-based anomaly scores using flattened encoder embeddings.

    Args:
        model: encoder model that maps [N_test, C2, T] → [N_test, C_out, H_out]
        features_training: latent training features [N_train, C1, H]
        testing: raw test signals [N_test, C2, T]
        R: number of nearest neighbors

    Returns:
        anomaly_score_embedding: np.ndarray, shape (N_test,)
            Euclidean distance between each test embedding and
            its selected 'memory' embedding (closest-to-median among R-NN)
    """
    device = next(model.parameters()).device
    model.eval()

    # 1) Flatten & normalize training features
    N_train = features_training.size(0)
    Z_train = features_training.reshape(N_train, -1).to(device)  # [N_train, C1*H]
    Z_train = F.normalize(Z_train, dim=1)

    # 2) Forward encoder on test set and flatten
    # z_test = model(testing.to(device))                # [N_test, C1, H]
    if feat_selection == 'z':
        z_test, _, _ = model.encode_to_proj(testing.to(device))
    elif feat_selection == 'h':
        _, z_test, _ = model.encode_to_proj(testing.to(device))
        
    z_test = z_test.reshape(z_test.size(0), -1)       # [N_test, C1*H]
    z_test = F.normalize(z_test, dim=1)

    # 3) Cosine distance matrix (dist = 1 - sim)
    sim = z_test @ Z_train.T                          # [N_test, N_train]
    dist_mat = 1.0 - sim

    # 4) Take R nearest neighbors efficiently
    dist_R, idx_R = torch.topk(dist_mat, k=R, dim=1, largest=False)

    # 5) Median distance & nearest-to-median neighbor
    med = dist_R.median(dim=1).values
    offset = (dist_R - med[:, None]).abs()
    choose = offset.argmin(dim=1)
    mem_idx = idx_R[torch.arange(idx_R.size(0), device=device), choose]

    # 6) Pull memory embeddings & compute Euclidean distance
    mem_emb = Z_train[mem_idx]
    anomaly_score_embedding = torch.linalg.norm(z_test - mem_emb, dim=1)

    return anomaly_score_embedding.cpu().numpy()


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


def coreset_pca_visualization(data, coreset_indices, full_path):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1])
    plt.scatter(principalComponents[coreset_indices, 0], principalComponents[coreset_indices, 1], marker='*')
    plt.legend(['Whole', 'Coreset'])
    plt.savefig(full_path, bbox_inches='tight')
    plt.show()


def filter_by_iforest(X, keep_ratio=0.95):
    clf = IsolationForest(contamination=1-keep_ratio, random_state=0)
    labels = clf.fit_predict(X)  # 1 = inlier, -1 = outlier
    keep_idx = np.where(labels == 1)[0]
    out_idx  = np.where(labels == -1)[0]
    return keep_idx, out_idx


def visualize_tsne_before_after(embeddings, keep_idx, full_path, perplexity=30, pca_dim=50, random_state=0):
    """
    Plot t-SNE before & after filtering in one figure.
    
    Parameters
    ----------
    embeddings : array-like, shape (N, p)
        High-dimensional data (e.g., SimCLR embeddings)
    keep_idx : array-like
        Indices of samples kept after filtering (e.g., 95% majority)
    perplexity : float, default=30
        t-SNE perplexity
    pca_dim : int, default=50
        PCA dimension reduction before t-SNE (for stability)
    random_state : int
        Random seed for reproducibility
    """
    X = np.asarray(embeddings, dtype=np.float32)

    # 1) PCA → t-SNE (for speed and stability)
    if X.shape[1] > pca_dim:
        Xp = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)
    else:
        Xp = X

    perpl = min(perplexity, max(5, (len(Xp)-1)//3))
    Y = TSNE(n_components=2, perplexity=perpl, init="pca",
             learning_rate="auto", random_state=random_state).fit_transform(Xp)

    # 2) Separate kept vs filtered points
    mask = np.zeros(len(X), dtype=bool)
    mask[np.asarray(keep_idx, dtype=int)] = True
    Y_keep, Y_out = Y[mask], Y[~mask]

    # 3) 95% contour for kept cluster (via KDE)
    kde = KernelDensity(bandwidth='scott', kernel='gaussian')
    kde.fit(Y_keep)
    x_min, y_min = Y_keep.min(0) - 0.05 * Y_keep.ptp(0)
    x_max, y_max = Y_keep.max(0) + 0.05 * Y_keep.ptp(0)
    xs, ys = np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    Xg, Yg = np.meshgrid(xs, ys)
    Z = np.exp(kde.score_samples(np.c_[Xg.ravel(), Yg.ravel()])).reshape(200, 200)
    dx, dy = (x_max - x_min)/199, (y_max - y_min)/199
    mass = Z * dx * dy
    order = np.argsort(mass.ravel())[::-1]
    cumsum = np.cumsum(mass.ravel()[order])
    idx95 = np.searchsorted(cumsum, 0.95)
    level = Z.ravel()[order[idx95]]

    # 4) Plot
    plt.figure(figsize=(7, 6))
    plt.scatter(Y_out[:,0], Y_out[:,1], c='red', marker='x', s=40, alpha=0.7, label='Filtered-out 5%')
    plt.scatter(Y_keep[:,0], Y_keep[:,1], c='blue', s=15, alpha=0.6, label='Kept 95%')
    plt.contour(Xg, Yg, Z, levels=[level], colors='black', linewidths=1.8, linestyles='--')
    plt.title("t-SNE: before & after filtering (95% contour)")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(full_path, bbox_inches='tight')
    plt.show()


def wasserstein_w2_sinkhorn(X, Y, reg=1e-1, max_iter=2000, tol=1e-9, return_plan=False):
    """
    Entropic-regularized 2-Wasserstein (approx). Much faster and memory friendlier.
    reg: entropy regularization (bigger = smoother/faster, more bias).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = len(X), len(Y)
    a = np.full(n, 1.0/n)
    b = np.full(m, 1.0/m)
    C = ot.dist(X, Y, metric='euclidean') ** 2

    G = ot.sinkhorn(a, b, C, reg=reg, numItermax=max_iter, stopThr=tol)
    W2 = np.sqrt((G * C).sum())
    return (W2, G) if return_plan else W2


def wasserstein_w2(X, Y, return_plan=False):
    """
    Exact 2-Wasserstein distance between two empirical measures in R^d.
    Unweighted, different sizes allowed (uniform weights implied).

    Parameters
    ----------
    X : (n, d) array
    Y : (m, d) array
    return_plan : bool
        If True, also return the optimal transport plan (n x m).

    Returns
    -------
    W2 : float
        The 2-Wasserstein distance.
    G  : (optional) ndarray
        Optimal transport plan.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    n, m = len(X), len(Y)

    a = np.full(n, 1.0/n)  # uniform weights
    b = np.full(m, 1.0/m)

    # Cost matrix = squared Euclidean distances
    C = ot.dist(X, Y, metric='euclidean') ** 2  # shape (n, m)

    # Exact EMD (LP)
    G = ot.emd(a, b, C)
    W2 = np.sqrt((G * C).sum())
    return (W2, G) if return_plan else W2


class RandJitter:       # small noise
    def __init__(self, sigma=0.02): self.sigma = sigma
    def __call__(self, x): return x + torch.randn_like(x) * self.sigma


class RandScaling:      # per-sequence amplitude scale
    def __init__(self, sigma=0.1): self.sigma = sigma
    def __call__(self, x):
        s = torch.randn(x.size(0), 1, device=x.device) * self.sigma + 1.0
        return x * s


class RandTimeShift:    # circular shift
    def __init__(self, max_shift=20): self.max_shift = max_shift
    def __call__(self, x): return torch.roll(x, shifts=random.randint(-self.max_shift, self.max_shift), dims=-1)


class RandCropResize:   # crop then resize back to 274
    def __init__(self, L=274, min_ratio=0.6, max_ratio=1.0):
        self.L, self.min_ratio, self.max_ratio = L, min_ratio, max_ratio
    def __call__(self, x):
        _, L = x.shape
        crop = max(8, int(L * random.uniform(self.min_ratio, self.max_ratio)))
        start = random.randint(0, L - crop)
        seg = x[:, start:start+crop].unsqueeze(0)
        return F.interpolate(seg, size=self.L, mode="linear", align_corners=False).squeeze(0)


class RandFreqMask:     # simple band-stop in rFFT
    def __init__(self, max_band=12, p=0.5): self.max_band, self.p = max_band, p
    def __call__(self, x):
        if random.random() > self.p: return x
        _, L = x.shape
        X = torch.fft.rfft(x, dim=-1)
        band = random.randint(1, min(self.max_band, X.size(-1)-1))
        s = random.randint(0, X.size(-1)-band-1)
        X[..., s:s+band] = 0
        return torch.fft.irfft(X, n=L, dim=-1).real


class Compose:
    def __init__(self, ops): self.ops = ops
    def __call__(self, x):
        for op in self.ops: x = op(x)
        return x

def aug_1d_univariate():
    return Compose([
        # RandCropResize(L=274, min_ratio=0.6, max_ratio=1.0),
        RandTimeShift(max_shift=10),
        RandScaling(sigma=0.01),
        RandJitter(sigma=0.01),
        # RandFreqMask(max_band=10, p=0.4),
    ])


def info_nce(z1, z2, tau=0.2):
    """
    z1, z2: (B, D) normalized projections (cosine sims).
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)            # (2B, D)
    sim = (z @ z.T) / tau                     # cosine sim because L2-normalized
    sim.fill_diagonal_(float('-inf'))         # avoid self-contrast
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)
    loss = -sim[torch.arange(2*B, device=z.device), pos] + torch.logsumexp(sim, dim=1)
    return loss.mean()


def consis_var_loss(z_fw: torch.Tensor, reduction: str = "sum", eps: float = 0.0):
    """
    z_fw: (B, C, R)  – first R columns of z (no flatten)
    Computes variance across batch (dim=0) for each (c,r), then reduces.
    """
    # center across batch and compute per-(c,r) variance
    # use unbiased=False to avoid 1/(B-1) blowup for small B
    var_cr = z_fw.var(dim=0, unbiased=False)       # (C, R)
    if eps > 0:
        var_cr = torch.clamp(var_cr, min=eps)      # optional floor to avoid degenerate grads
    if reduction == "sum":
        return var_cr.sum()
    elif reduction == "mean":
        return var_cr.mean()
    else:
        return var_cr   # (C, R) if you want to inspect it



def per_series_zscore(X):  # X: [N,1,274] tensor
    mu = X.mean(dim=-1, keepdim=True)
    sd = X.std(dim=-1, keepdim=True).clamp_min(1e-6)
    return (X - mu) / sd


def numpy_per_series_zscore(X):
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    return (X - mean) / (std + 1e-6)


# --- differentiable Sinkhorn W2 (entropic OT) ---
def sinkhorn_w2_torch(X, Y, eps=0.05, n_iters=50):
    """
    X: (n, d), Y: (m, d). Returns approx 2-Wasserstein distance (not squared).
    Entropic OT with uniform weights; differentiable.
    """
    n, m = X.size(0), Y.size(0)
    if n == 0 or m == 0:
        # edge case: if a pristine group is empty in the batch
        return X.new_tensor(0.0)

    a = X.new_full((n,), 1.0/n)               # uniform weights
    b = Y.new_full((m,), 1.0/m)

    # Ground cost: squared Euclidean distances
    C = torch.cdist(X, Y, p=2)**2             # (n, m)

    # Sinkhorn iterations in scaling form
    K = torch.exp(-C / eps)                   # (n, m)
    u = a.clone()
    v = b.clone()
    for _ in range(n_iters):
        Ku = K @ v + 1e-9
        u = a / Ku
        KTu = K.t() @ u + 1e-9
        v = b / KTu

    # Transport plan
    # P = diag(u) K diag(v)
    P = (u[:, None] * K) * v[None, :]
    cost = (P * C).sum()
    return torch.sqrt(cost + 1e-12)


def softmin(values, alpha=10.0):
    """Smooth min:  -1/alpha * logsumexp(-alpha * values). Keeps gradients."""
    return -torch.logsumexp(-alpha * values, dim=0) / alpha


def vicreg_loss(p1, p2, sim_w=25, var_w=25, cov_w=5, gamma=0.9, eps=1e-4):
    # invariance
    inv = F.mse_loss(p1, p2)

    # variance (per-dim std; penalize if < gamma)
    def var_term(z):
        z = z - z.mean(dim=0, keepdim=True)
        std = z.std(dim=0, unbiased=False) + eps
        return torch.mean(F.relu(gamma - std))

    # covariance (off-diagonal)
    def cov_term(z):
        z = z - z.mean(dim=0, keepdim=True)
        N = z.size(0)
        cov = (z.T @ z) / (N - 1)
        off = cov - torch.diag(torch.diag(cov))
        return (off**2).sum() / p1.size(1)

    var = var_term(p1) + var_term(p2)
    cov = cov_term(p1) + cov_term(p2)

    return sim_w * inv + var_w * var + cov_w * cov, inv, var, cov


def cosine_schedule(base, min_val, t, T):
    # t in [0, T-1]
    return min_val + 0.5 * (base - min_val) * (1.0 + np.cos(np.pi * t / max(1, T-1)))


def warmup_factor(epoch, warmup_epochs):
    if warmup_epochs <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / float(warmup_epochs))


def get_fft_max_amplitude(X_test, seg_start=150, seg_end=200, L=1000):
    # 1. Find the max index within the [150:200] range for all rows at once
    # argmax returns index 0-49, so we add 150 to get absolute indices
    relative_max_pos = np.argmax(X_test[:, seg_start:seg_end], axis=1)
    absolute_max_pos = relative_max_pos + seg_start
    
    # 2. Define window boundaries (30 back, 30 forward)
    starts = absolute_max_pos - 30
    ends = absolute_max_pos + 31 # +31 because slice is exclusive
    
    # 3. Extract segments
    # Since segments are the same length (61), we can use advanced indexing 
    # to create a 2D matrix of segments [N, 61]
    rows = np.arange(X_test.shape[0])[:, None]
    cols = starts[:, None] + np.arange(61)
    
    # Clip indices to handle edge cases where the window might exceed 0 or 274
    cols = np.clip(cols, 0, X_test.shape[1] - 1)
    segments = X_test[rows, cols]
    
    # 4. Perform FFT across the rows with padding n=L
    # We use rfft (Real FFT) since your input is likely real; it's faster
    # and only returns the positive frequencies (0 to L/2)
    fft_results = np.fft.rfft(segments, n=L, axis=1)
    
    # 5. Calculate amplitudes and find the maximum for each row
    amplitudes = np.abs(fft_results)
    anomaly_scores = np.max(amplitudes, axis=1)
    
    return anomaly_scores


