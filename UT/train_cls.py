from utlis import *
from models import *

device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} device')

# Hyperparameters
depth = 320
width = 30
batch_size = 64
learning_rate = 1e-4
adam_weight_decay = 1e-5
scheduler_factor = 0.5
scheduler_patience = 20
epochs = 100

# Get data
current_path = os.getcwd()
training_file_path = os.path.join(current_path, 'data\\training_cls_1346.csv')
validation_file_path = os.path.join(current_path, 'data\\validation_cls_25.csv')
training_data = np.genfromtxt(training_file_path, delimiter=',').astype('f4')
np.random.shuffle(training_data)
training_data = torch.from_numpy(training_data).to(device)
validation_data = torch.from_numpy(np.genfromtxt(validation_file_path, delimiter=',').astype('f4')).to(device)

training_X = training_data[:, 0:depth]
training_y = training_data[:, depth].type(torch.LongTensor).to(device)
validation_X = validation_data[:, 0:depth]
validation_y = validation_data[:, depth].type(torch.LongTensor).to(device)

# Initialization
selection = 'MLPwAttn'
if selection == 'MLP':
    model = MLP().to(device)
    pth_name = 'model_para\\model.pth'
elif selection == 'MLPwAttn':
    model = MLPwAttn(input_dim=depth, embed_dim=8).to(device)
    pth_name = 'model_para\\model_attn.pth'
    training_X = training_X * 127.5 + 127.5
    training_X = training_X.type(torch.LongTensor).to(device)
    validation_X = validation_X * 127.5 + 127.5
    validation_X = validation_X.type(torch.LongTensor).to(device)

loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=adam_weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=scheduler_factor, patience=scheduler_patience)
batch_start = torch.arange(0, training_X.shape[0], batch_size)

# Training
model.train()
for epoch in range(epochs):
    for start in batch_start:
        model.train()
        X = training_X[start:start+batch_size, :]
        y = training_y[start:start+batch_size]

        pred_prob, _ = model(X)
        training_loss = loss_function(pred_prob, y)
    
        training_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    model.eval()
    with torch.no_grad():
        validation_pred_prob, _ = model(validation_X)
        validation_loss = loss_function(validation_pred_prob, validation_y)
        scheduler.step(validation_loss)

    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, training loss: {training_loss}, validation loss: {validation_loss}')

torch.save(model.state_dict(), pth_name)

model.eval()
training_pred_prob, _ = model(training_X)
validation_pred_prob, _ = model(validation_X)

training_accuracy = (training_pred_prob.argmax(1) == training_y).type(torch.float).sum().item() / training_X.shape[0]
validation_accuracy = (validation_pred_prob.argmax(1) == validation_y).type(torch.float).sum().item() / validation_X.shape[0]
print(f'Training accuracy: {training_accuracy}, validation accuracy: {validation_accuracy}')

