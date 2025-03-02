import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dataset loading and transformation
transform = transforms.Compose([
    transforms.Resize((200, 300)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root="path_to_caltech256", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Plotting sample images
def plot_samples(dataloader, num_samples=5):
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, num_samples, figsize=(12, 4))
    for i in range(num_samples):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].set_title(f"Class {labels[i]}")
        axes[i].axis("off")
    plt.show()

plot_samples(dataloader)

# Neural Network Definition
class FeedforwardNN(nn.Module):
    def _init_(self, input_size, hidden_layers, hidden_units, output_size, activation_fn):
        super(FeedforwardNN, self)._init_()
        layers = []
        in_features = input_size
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_units))
            layers.append(activation_fn())
            in_features = hidden_units
        layers.append(nn.Linear(in_features, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# Hyperparameter configurations to test
configurations = [
    {"epochs": 5, "hidden_layers": 3, "hidden_units": 64, "learning_rate": 1e-3, "optimizer_type": 'adam', "batch_size": 32, "activation_fn": nn.ReLU},
    {"epochs": 10, "hidden_layers": 4, "hidden_units": 128, "learning_rate": 1e-4, "optimizer_type": 'sgd', "batch_size": 64, "activation_fn": nn.ReLU},
    {"epochs": 5, "hidden_layers": 5, "hidden_units": 32, "learning_rate": 1e-3, "optimizer_type": 'momentum', "batch_size": 16, "activation_fn": nn.Sigmoid},
    {"epochs": 10, "hidden_layers": 3, "hidden_units": 64, "learning_rate": 1e-3, "optimizer_type": 'rmsprop', "batch_size": 32, "activation_fn": nn.ReLU},
    {"epochs": 5, "hidden_layers": 4, "hidden_units": 128, "learning_rate": 1e-4, "optimizer_type": 'adam', "batch_size": 64, "activation_fn": nn.ReLU},
]

# Model, loss, and optimizer initialization
def get_optimizer(model, optimizer_type, learning_rate):
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'momentum':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == 'nesterov':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)

# Training Loop
def train(model, dataloader, criterion, optimizer, epochs):
    best_val_accuracy = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Track the best model based on validation accuracy
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_model = model.state_dict()

    return best_model, best_val_accuracy

# Evaluation on test set
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total

# Test different configurations and select the best one
best_accuracy = 0
best_config = None
best_test_accuracy = 0
best_model = None

for config in configurations:
    print(f"\nTesting configuration: {config}")
    
    model = FeedforwardNN(input_size=200*300*3, 
                          hidden_layers=config['hidden_layers'], 
                          hidden_units=config['hidden_units'], 
                          output_size=256, 
                          activation_fn=config['activation_fn'])
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config['optimizer_type'], config['learning_rate'])
    
    # Train the model
    best_model_state, val_accuracy = train(model, dataloader, criterion, optimizer, config['epochs'])
    
    # Load the best model after training
    model.load_state_dict(best_model_state)

    # Evaluate on validation set
    test_dataset = datasets.ImageFolder(root="path_to_caltech256_test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    test_accuracy = evaluate(model, test_loader)

    print(f"Test Accuracy: {test_accuracy:.2f}%")

    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        best_config = config
        best_model = model

# Final Results
print(f"\nBest configuration: {best_config}")
print(f"Best Test Accuracy: {best_test_accuracy:.2f}%")
