import torch
from torch import nn

# 定义神经网络Network


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()  # 继承nn.Module的属性和方法
        # 线性层1，输入层和隐藏层之间的线性层，输入维度为784，输出维度为256
        self.layer1 = nn.Linear(784, 256)
        # 线性层2，隐藏层和输出层之间的线性层，输入维度为256，输出维度为10
        self.layer2 = nn.Linear(256, 10)

# 定义前向传播函数

    def forward(self, x):
        x = x.view(-1, 784)  # 将输入张量x展平为二维张量，-1表示自动计算维度大小 view() 函数用于改变张量的形状
        x = self.layer1(x)  # 将输入张量x传递给线性层1，得到输出张量x
        x = torch.relu(x)  # 对输出张量x应用ReLU激活函数
        return self.layer2(x)  # 将激活后的输出张量x传递给线性层2，得到最终的输出张量

# 没有直接定义softmax层是因为后面会使用 nn.CrossEntropyLoss() 函数即交叉熵损失函数，该函数内部已经包含了softmax层，该函数会自动对输出进行softmax操作
