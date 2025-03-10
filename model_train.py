import torch
from torch import nn
from torch import optim
from model import Network
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 数据预处理：将图像转为灰度张量格式
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为单通道灰度图
        transforms.ToTensor()  # 转换为PyTorch张量
    ])

    # 加载MNIST训练集和测试集
    train_dataset = datasets.ImageFolder(
        root='./mnist_train', transform=transform)  # 从指定目录加载训练集

    test_dataset = datasets.ImageFolder(
        root='./mnist_test', transform=transform)  # 从指定目录加载测试集
    # 打印训练数据集大小
    print("训练集长度: ", len(train_dataset))
    # 打印测试数据集大小
    print("测试集长度: ", len(test_dataset))

    # 创建数据加载器（仅使用训练集）
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True)  # 批量大小64，打乱顺序

    print("训练集加载器长度: ", len(train_loader))

    # 60000个训练数据，如果每个小批量，读入64个样本，那么60000个数据会被分成938组
    # 计算938x64=60032，这说明最后一组不够64个数据

    # 遍历训练DataLoader 的前几个批次
    for batch_idx, (data, label) in enumerate(train_loader):

        print("batch_idx: ", batch_idx)

        print("data.shape: ", data.shape)

        print("label.shape: ", label.shape)

        print(label)

    # 初始化神经网络模型
    model = Network()

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters())  # 使用Adam优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数 用于分类问题

    # 训练循环（10个epoch）
    for epoch in range(10):
        # 遍历训练数据
        for batch_idx, (data, label) in enumerate(train_loader):
            # 前向传播
            output = model(data)

            # 计算损失
            loss = criterion(output, label)

            # 反向传播
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数
            optimizer.zero_grad()  # 梯度清零

            # 每100个batch打印训练进度
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), 'mnist.pth')  # 保存模型参数到文件
