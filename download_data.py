import os
import gzip
import shutil
import requests

# 创建 MNIST_data 文件夹
mnist_dir = 'MNIST_data'
if not os.path.exists(mnist_dir):
    os.makedirs(mnist_dir)

# 下载并解压文件


def download_and_extract(url, file_name):
    print(f"Downloading {file_name}...")
    response = requests.get(url)
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print(f"Extracting {file_name}...")
    with gzip.open(file_name, 'rb') as f_in:
        output_file = os.path.join(
            mnist_dir, file_name[:-3])  # 解压到 MNIST_data 文件夹
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(file_name)  # 删除压缩文件


# MNIST 文件列表
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

# 下载并解压所有文件
base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
for file in files:
    url = base_url + file
    download_and_extract(url, file)

print("下载和解压完成!")
