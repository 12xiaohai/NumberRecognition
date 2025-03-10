import numpy as np
import struct
from PIL import Image
import os

# 初始化并打印开始提示
print("处理开始！")

# 处理图像数据文件
data_file = r'./MNIST_data/train-images.idx3-ubyte'
# 原始文件大小包含16字节文件头，实际数据需要减去头大小
data_file_size = 47040016
data_file_size = str(data_file_size - 16) + 'B'  # 计算实际数据大小
# 解析二进制文件头（魔数、图片数量、尺寸信息）
data_buf = open(data_file, 'rb').read()
magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)
# 读取图像数据并转换为numpy数组（格式：数量×通道×高度×宽度）
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

# 处理标签数据文件
label_file = r'MNIST_data/train-labels.idx1-ubyte'
# 原始文件包含8字节文件头，标签数据需要减去头大小
label_file_size = 60008
label_file_size = str(label_file_size - 8) + 'B'  # 计算实际标签数量
# 解析标签文件
label_buf = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size,
                            label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

# 创建按类别分类的存储目录
datas_root = 'mnist_train'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

# 为0-9每个数字创建单独目录
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

# 保存处理进度提示
print(" 正在处理图像...")

# 遍历所有样本进行保存
for ii in range(numLabels):
    # 每处理1000张输出进度
    if ii % 1000 == 0:
        print(f"处理 {ii}/{numLabels} 图像...")
    # 将numpy数组转换为PIL图像对象
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])
    label = labels[ii]
    # 按标签分类保存图片
    file_name = datas_root + os.sep + \
        str(label) + os.sep + 'mnist_train_' + str(ii) + '.png'
    img.save(file_name)

print("处理完成！")
