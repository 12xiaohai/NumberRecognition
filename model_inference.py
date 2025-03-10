from model import Network  # 导入自定义神经网络模型类
from torchvision import transforms  # 导入torchvision图像变换模块
from torchvision import datasets  # 导入torchvision数据集模块
import torch  # 导入PyTorch

if __name__ == '__main__':
    # 定义图像预处理流程：转灰度图 -> 转张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 加载测试数据集（从指定目录）并应用预处理
    test_dataset = datasets.ImageFolder(
        root='./mnist_test', transform=transform)
    # 打印测试集样本数量
    print("测试集长度: ", len(test_dataset))

    # 初始化神经网络模型
    model = Network()
    # 从保存的文件中加载模型的状态字典
    model.load_state_dict(torch.load('mnist.pth'))

    right = 0  # 正确分类计数器

    # 遍历测试数据集
    for i, (x, y) in enumerate(test_dataset):
        # 前向传播：添加批次维度并计算模型输出
        output = model(x.unsqueeze(0))
        # 取最高得分的索引作为预测标签
        predict = output.argmax(1).item()
        if predict == y:
            right += 1  # 预测正确时增加计数器
        else:
            # 获取错误样本的文件路径
            img_path = test_dataset.samples[i][0]
            # 打印错误案例详细信息
            print(
                f"错误结果: 预测 = {predict} 实际 = {y} img_path = {img_path}")

    # 获取总样本数
    sample_num = len(test_dataset)
    # 计算准确率（正确预测数除以总样本数）
    acc = right * 1.0 / sample_num
    # 打印测试准确率
    print("测试准确率 = %d / %d = %.31f" % (right, sample_num, acc))
