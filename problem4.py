# 2. Implement backpropagation in NumPy or PyTorch using basic linear algebra operations. (e.g. You are not allowed to
# use auto-grad, built-in optimizer, model, etc. in this step. You can use library functions for data loading,
# processing, etc.). Evaluate your implementation on MNIST dataset, report test errors and learning curve.
# (10 pts)

import numpy as np 

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(self.hidden_size, self.input_size)
        self.W2 = np.random.randn(self.output_size, self.hidden_size)

    def forward(self, x):
        z1 = np.dot(self.W1, x)
        a1 = self.sigmoid(z1)
        z2 = np.dot(self.W2, a1)
        y_hat = self.softmax(z2)
        return z1, a1, z2, y_hat
    
    def sigmoid(self, z):
        # Sigmoid activation function
        return 1 / (1 + np.exp(-z))
    
    def softmax(self, z):
        # Softmax activation function
        return np.exp(z) / np.sum(np.exp(z))

    def cross_entropy_loss(self, y, y_hat):
        # Compute the cross entropy loss
        return - np.sum(y * np.log(y_hat))

    def backward(self, x, y, y_hat, a1):
        a1 = a1.reshape(-1, 1)
        x = x.reshape(-1, 1)
        d2 = (y_hat - y).reshape(-1, 1)
        dL_dW2 = np.dot(d2, a1.reshape(1, -1))
        d1 = np.dot(self.W2.T, d2) * (a1*(1 - a1))
        dL_dW1 = np.dot(d1, x.T)
        return dL_dW1, dL_dW2

    def train(self, X, y, epochs, batch_size, num_datapoints):
        losses = []
        num_batches = X.shape[0] // batch_size
        for epoch in range(epochs):
            epoch_loss = 0
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                X_batch = X[start_idx:end_idx]
                Y_batch = y[start_idx:end_idx]
                X_batch = X_batch.reshape(batch_size, -1)
                dL_dW1, dL_dW2 = 0, 0
                for j in range(batch_size):
                    x_batch = X_batch[j]
                    y_batch = np.zeros(10)
                    y_batch[Y_batch[j]] = 1
                    _, a1, _, y_hat = self.forward(x_batch)
                    epoch_loss += self.cross_entropy_loss(y_batch, y_hat)
                    new_dL_dW1, new_dL_dW2 = self.backward(x_batch, y_batch, y_hat, a1)
                    dL_dW1 += new_dL_dW1
                    dL_dW2 += new_dL_dW2
                self.W1 -= self.learning_rate * dL_dW1/batch_size
                self.W2 -= self.learning_rate * dL_dW2/batch_size

            epoch_loss /= num_batches
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss}')
            losses.append(epoch_loss)
        return losses
    
    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            x = X[i].reshape(-1, 1)
            _, _, _, y_hat = self.forward(x)
            y_pred.append(np.argmax(y_hat))
        return np.array(y_pred)

# Load the MNIST dataset
import torch 
from torchvision import datasets, transforms

# Download and load the training data
train_ds = datasets.MNIST(root="./data", train=True, download=True)
test_ds = datasets.MNIST(root="./data", train=False, download=True)
# Extract features and labels
X_train = train_ds.data.numpy()
y_train = train_ds.targets.numpy()
X_train = X_train / 255.0
y_train = y_train.astype(np.uint8)

X_test = test_ds.data.numpy()
y_test = test_ds.targets.numpy()
X_test = X_test / 255.0
y_test = y_test.astype(np.uint8)

# Initialize neural network
nn = NeuralNetwork(784, 200, 10, 0.01)

# Train the neural network
# losses = nn.train(X_train, y_train, 10, 32, 60000)

# # Plot the learning curve
import matplotlib.pyplot as plt
# plt.plot(losses)
# plt.xlabel('Epoch')
# plt.ylabel('Cross Entropy Loss')
# plt.title('Learning Curve d1=200, lr=0.01, batch_size=32')
# plt.savefig('learning_curve.pdf')
# plt.clf()

# # Plot Test Error
# predictions = nn.predict(X_test)
# test_error = np.mean(predictions != np.argmax(y_test, axis=0))
# print('Test Error: {}'.format(test_error))

import torch.nn as nn
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.flat(x)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return nn.functional.softmax(x, dim=1)

# Initialize the Model
model = Net(784, 200, 10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
losses = []
# MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Initialize the weights
weight_init = "random"
if weight_init == "zeros":
    model.fc1.weight.data = torch.zeros(200, 784)
    model.fc2.weight.data = torch.zeros(10, 200)
elif weight_init == "random":
    model.fc1.weight.data = torch.randn(200, 784)
    model.fc2.weight.data = torch.randn(10, 200)

# Training the model
num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    losses.append(loss.item())

# Plot the learning curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy Loss')
plt.title('Learning Curve d1=200, lr=0.01, batch_size=32')
plt.savefig('learning_curve_pytorch_random.pdf')
plt.clf()

# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total}%')
