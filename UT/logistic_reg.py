from utlis import *
from sklearn.linear_model import LogisticRegression

depth = 320
width = 30

# Get data
current_path = os.getcwd()
training_file_path = os.path.join(current_path, 'training.csv')
validation_file_path = os.path.join(current_path, 'validation.csv')
training_data = np.genfromtxt(training_file_path, delimiter=',').astype('f4')
validation_data = np.genfromtxt(validation_file_path, delimiter=',').astype('f4')
training_X = training_data[:, 0:depth]
training_y = training_data[:, depth]
validation_X = validation_data[:, 0:depth]
validation_y = validation_data[:, depth]

# Fit
clf = LogisticRegression(random_state=0, max_iter=500).fit(training_X, training_y)
training_accuracy = clf.score(training_X, training_y)
validation_accuracy = clf.score(validation_X, validation_y)
print(f'Training accuracy: {training_accuracy}, validation accuracy: {validation_accuracy}')

# Plot
whole_file_path = os.path.join(current_path, 'whole.csv')
whole_X = np.genfromtxt(whole_file_path, delimiter = ',')
whole_PBRA = np.concatenate((whole_X[0:900, :], whole_X[1800:2700, :], whole_X[900:1800, :], 
                             whole_X[2700:3600, :], whole_X[4500:5400, :], whole_X[3600:4500, :],
                             whole_X[8100:10800, :], whole_X[6300:7200, :]))  # 1, 3, 2, 4, 6, 5, 10-12, 8
whole_pred = clf.predict(whole_PBRA)

k = 0
fig, axes = plt.subplots(nrows = 4, ncols = 3)
for ax in axes.flat:
    score = whole_pred[k*width**2:(k+1)*width**2].reshape(width, width)
    im = ax.imshow(score, vmin=0, vmax=1)
    k += 1
    if k == 10:
        break
axes[3, 1].axis('off')
axes[3, 2].axis('off')
fig.colorbar(im, ax=axes.ravel().tolist())
plt.show()
