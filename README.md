# NetworkLearning

## 快速开始

```
pip install -r requirements.txt
python download_data.py
python parse_train_images_labels.py
python parse_t10k_images_labels.py
python model_train.py
python model_inference.py
```

## 基于PyTorch的MNIST手写数字识别

这是人工智能（AI）和深度学习领域的"Hello World"程序，专为学习目的编写。

本仓库包含使用PyTorch实现MNIST手写数字识别的完整神经网络项目，包含训练和推理脚本以及数据集处理工具。

### 项目结构

```
├── MNIST_data                 # 原始MNIST数据集 运行download_data.py即可得到该文件夹
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
├── mnist_train               # 处理后的训练集
│   ├── 0
│   ├── 1
│   ├── ...
│   ├── 9
├── mnist_test                # 处理后的测试集
│   ├── 0
│   ├── 1
│   ├── ...
│   ├── 9
├── download_data.py        # 下载原始MNIST数据集
├── model_inference.py        # 模型推理脚本
├── model_train.py            # 模型训练脚本
├── parse_t10k_images_labels.py  # 测试集处理脚本
├── parse_train_images_labels.py # 训练集处理脚本
```

### 脚本说明

#### download_data.py

功能：

- 下载原始MNIST数据集并保存到MNIST_data文件夹中

#### parse_train_images_labels.py
功能：
- 从.idx3-ubyte文件读取MNIST训练图像
- 解析图像并按数字标签分类保存到mnist_train目录

#### parse_t10k_images_labels.py
功能：
- 从.idx1-ubyte文件读取MNIST测试标签
- 解析标签并保存对应图像到mnist_test目录

#### model_train.py
功能：
- 从./mnist_train和./mnist_test加载数据集
- 应用图像转换（灰度化、张量转换）
- 使用Adam优化器和交叉熵损失训练神经网络
- 保存训练好的模型到mnist.pth

#### model_inference.py
功能：
- 加载测试集和预训练模型
- 执行推理并打印错误案例
- 计算并输出测试准确率

****

### 使用方法

1. 准备数据集：
   - 下载MNIST原始数据集并保存到MNIST_data目录
   
     ```shell
     python download_data.py
     ```
   
     
   
   - 运行数据处理脚本：
     ```shell
     python parse_train_images_labels.py
     python parse_t10k_images_labels.py
     ```
   
2. 训练模型：
   ```shell
   python model_train.py
   ```
   训练过程将显示损失值变化，最终模型保存为mnist.pth

3. 执行推理：
   ```shell
   python model_inference.py
   ```
   输出结果包含：
   - 错误预测案例（预测值/实际值/图像路径）
   - 最终测试准确率

### 测试结果（97.99%）

完整测试输出显示：
```txt
测试准确率 = 9799 / 10000 = 0.9799
```
共223个错误案例，详细错误日志包含预测值、实际值和对应图像路径。

### 依赖项

- Python 3.9
- PyTorch
- torchvision
- Pillow
- numpy
- ... ... ... ...

安装命令：
```shell
pip install -r requirements.txt
```

### 致谢

- MNIST数据集由Yann LeCun提供：[官网链接](https://yann.lecun.com/exdb/mnist/)
- 基于PyTorch开源机器学习框架开发

