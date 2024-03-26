import torch
import torch.nn as nn
import torch.optim as optim

# Define the model architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 26)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc2(x)
        return x

# Instantiate the model
model = Net()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(5):
    for data, target in zip(train_data, train_labels):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluate the model
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in zip(test_data, test_labels):
        output = model(data)
        loss = criterion(output, target)
        test_loss += loss.item()
        _, predicted = torch.max(output, 1)
        if predicted == target:
            correct += 1

test_loss /= len(test_data)

print(f'Test accuracy: {1
