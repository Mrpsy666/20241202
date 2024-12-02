import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import BPNet, LeNet
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_BP():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = BPNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建 TensorBoard 写入器
    writer = SummaryWriter('logs/BP')

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 记录训练集损失
        writer.add_scalar('Train Loss', train_loss / len(train_loader), epoch)

    # 评估BP网络在测试集上的性能
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 记录测试集准确率
    writer.add_scalar('Test Accuracy', 100 * correct / total, num_epochs)

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    torch.save(model.state_dict(), 'BPNet.pth')
    writer.close()


def train_LeNet():
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 训练Lenet网络
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 创建 TensorBoard 写入器
    writer = SummaryWriter('logs/LeNet')

    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch"):
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 记录训练集损失
        writer.add_scalar('Train Loss', train_loss / len(train_loader), epoch)

    # 评估Lenet网络在测试集上的性能
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 记录测试集准确率
    writer.add_scalar('Test Accuracy', 100 * correct / total, num_epochs)

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
    torch.save(model.state_dict(), 'LeNet.pth')
    writer.close()

if __name__ == '__main__':
    print("Train BP Model")
    train_BP()
    print("Train LeNet Model")
    train_LeNet()
    print("Done")