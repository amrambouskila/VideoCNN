import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# Data Preprocessing and Loading
class VideoDataset(data.Dataset):
    def __init__(self, video_paths, labels, transform=None, sequence_length=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):
        video_path = self.video_paths[index]
        label = self.labels[index]
        frames = self._load_video(video_path)
        if self.transform:
            frames = self.transform(frames)
        return frames, label

    def _load_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (112, 112))
            frames.append(frame)
        cap.release()
        frames = np.stack(frames, axis=0)
        if len(frames) < self.sequence_length:
            pad_size = self.sequence_length - len(frames)
            padding = np.zeros((pad_size, 112, 112, 3), dtype=np.uint8)
            frames = np.concatenate((frames, padding), axis=0)
        frames = np.transpose(frames, (3, 0, 1, 2))  # (C, D, H, W)
        return frames


# Define the CNN model
class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv3d(in_channels, growth_rate, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(growth_rate),
            nn.ReLU(inplace=True)
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, 1))
            features.append(out)
        return torch.cat(features, 1)


class InceptionModule3D(nn.Module):
    def __init__(self, in_channels, out1x1, out3x3red, out3x3, out5x5red, out5x5, out_pool):
        super(InceptionModule3D, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, out1x1, kernel_size=1),
            nn.BatchNorm3d(out1x1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, out3x3red, kernel_size=1),
            nn.BatchNorm3d(out3x3red),
            nn.ReLU(inplace=True),
            nn.Conv3d(out3x3red, out3x3, kernel_size=3, padding=1),
            nn.BatchNorm3d(out3x3),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, out5x5red, kernel_size=1),
            nn.BatchNorm3d(out5x5red),
            nn.ReLU(inplace=True),
            nn.Conv3d(out5x5red, out5x5, kernel_size=3, padding=1),
            nn.BatchNorm3d(out5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm3d(out_pool),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        return torch.cat([branch1, branch2, branch3, branch4], 1)


class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class Attention3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(Attention3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv3 = nn.Conv3d(out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv1(x)
        attention = self.conv2(attention)
        attention = self.conv3(attention)
        attention = self.sigmoid(attention)
        return x * attention


class Advanced3DCNN(nn.Module):
    def __init__(self, num_classes, growth_rate=12, num_layers=4, reduction=16):
        super(Advanced3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, growth_rate * 2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(growth_rate * 2)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

        # DenseBlock
        self.dense_block = DenseBlock3D(growth_rate * 2, growth_rate, num_layers)
        self.transition = self._make_transition(growth_rate * (2 + num_layers), growth_rate * 2)

        # Inception
        self.inception = InceptionModule3D(growth_rate * 2, 64, 96, 128, 16, 32, 32)

        # SE Layer
        self.se = SELayer3D(growth_rate * 2)

        # Attention
        self.attention = Attention3D(growth_rate * 2, growth_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(growth_rate * 2 * 4 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def _make_transition(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.dense_block(x)
        x = self.transition(x)

        x = self.inception(x)

        x = self.se(x)

        x = self.attention(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Function to plot metrics
def plot_metrics(train_losses, val_losses, val_accuracies, test_loss, test_accuracy):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.bar(['Test'], [test_loss], label='Test Loss')
    plt.bar(['Test'], [test_accuracy], label='Test Accuracy')
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title('Test Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()


# Training and evaluation functions
def train(model, trainloader, validloader, testloader, epochs=10, learning_rate=0.001, weight_decay=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs =
