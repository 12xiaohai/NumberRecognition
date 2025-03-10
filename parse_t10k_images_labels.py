import numpy as np
import struct
from PIL import Image
import os

print("处理开始！")

# 处理图像数据文件（t10k-images-idx3-ubyte格式）
# --------------------------------------------------
data_file = r'./MNIST_data/t10k-images-idx3-ubyte'

data_file_size = 7840016
data_file_size = str(data_file_size - 16) + 'B'  # 减去文件头16字节
data_buf = open(data_file, 'rb').read()

# 解析二进制文件头（魔数、图片数量、行数、列数）
magic, numImages, numRows, numColumns = struct.unpack_from(
    '>IIII', data_buf, 0)

# 读取所有图像数据并转换为numpy数组（格式：数量×通道×高度×宽度）
datas = struct.unpack_from(
    '>' + data_file_size, data_buf, struct.calcsize('>IIII'))
datas = np.array(datas).astype(np.uint8).reshape(
    numImages, 1, numRows, numColumns)

# 处理标签数据文件（t10k-labels-idx1-ubyte格式）
# --------------------------------------------------
label_file = r'./MNIST_data/t10k-labels-idx1-ubyte'

label_file_size = 10008
label_file_size = str(label_file_size - 8) + 'B'
label_buf = open(label_file, 'rb').read()
magic, numLabels = struct.unpack_from('>II', label_buf, 0)
labels = struct.unpack_from('>' + label_file_size,
                            label_buf, struct.calcsize('>II'))
labels = np.array(labels).astype(np.int64)

# 创建分类存储目录
datas_root = 'mnist_test'
if not os.path.exists(datas_root):
    os.mkdir(datas_root)

# 为每个数字类别（0-9）创建单独目录
for i in range(10):
    file_name = datas_root + os.sep + str(i)
    if not os.path.exists(file_name):
        os.mkdir(file_name)

print("正在处理图像...")

# 遍历所有图像并保存为PNG格式
for ii in range(numLabels):
    # 每处理1000张输出进度
    if ii % 1000 == 0:
        print(f"处理 {ii}/{numLabels} 图像...")

    # 将numpy数组转换为PIL图像对象（28x28像素）
    img = Image.fromarray(datas[ii, 0, 0:28, 0:28])

    # 获取对应标签并保存到对应类别目录
    label = labels[ii]
    file_name = datas_root + os.sep + \
        str(label) + os.sep + 'mnist_test_' + str(ii) + '.png'
    img.save(file_name)

print("处理完成！")
