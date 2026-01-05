from utlis import *
from models import *

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

depth = 320
current_path = os.getcwd()
whole_file_path = os.path.join(current_path, 'data\\whole.csv')
whole_X = np.genfromtxt(whole_file_path, delimiter = ',').astype('f4')
whole_PBRA = np.concatenate((whole_X[0:900, :], whole_X[1800:2700, :], whole_X[900:1800, :], 
                             whole_X[2700:3600, :], whole_X[4500:5400, :], whole_X[3600:4500, :],
                             whole_X[8100:10800, :], whole_X[6300:7200, :]))  # 1, 3, 2, 4, 6, 5, 10-12, 8
whole_PBRA =torch.from_numpy(whole_PBRA).to(device)

selection = 'MLPwAttn'
if selection == 'MLP':
    model = MLP().to(device)
    model.load_state_dict(torch.load('model_para\\model.pth'))
elif selection == 'MLPwAttn':
    model = MLPwAttn(input_dim=depth, embed_dim=8).to(device)
    model.load_state_dict(torch.load('model_para\\model_attn.pth'))
    whole_PBRA = whole_PBRA * 127.5 + 127.5
    whole_PBRA = whole_PBRA.type(torch.LongTensor).to(device)

model.eval()
whole_pred_prob, attn_weight = model(whole_PBRA)
whole_pred = whole_pred_prob.argmax(1)

# Plot
k = 0
width = 30
fig, axes = plt.subplots(nrows = 4, ncols = 3)
for ax in axes.flat:
    score = whole_pred[k*width**2:(k+1)*width**2].view(width, width).cpu().detach().numpy()
    im = ax.imshow(score, vmin=0, vmax=1)
    k += 1
    if k == 10:
        break
axes[3, 1].axis('off')
axes[3, 2].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()

plt.matshow(attn_weight[450, :, :].view(depth, depth).cpu().detach().numpy())
plt.colorbar()
plt.show()

plt.matshow(attn_weight[3150, :, :].view(depth, depth).cpu().detach().numpy())
plt.colorbar()
plt.show()
