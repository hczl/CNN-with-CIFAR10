import torch
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import numpy as np
import gzip

from CNN_MNIST import cnn_train_test
from KNN import knn_train_test
from SVM import svm_train_test
from processing import CustomDataset, flatten_data


# 解压缩MNIST数据集文件
def extract_mnist_images(filename, num_images):
    with gzip.open(filename, 'rb') as f:
        f.read(16)  # 跳过文件头
        buf = f.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 1, 28, 28)
    return data


def extract_mnist_labels(filename, num_labels):
    with gzip.open(filename, 'rb') as f:
        f.read(8)  # 跳过文件头
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels


# 加载MNIST数据集
train_images = extract_mnist_images('train-images-idx3-ubyte.gz', 60000)
train_labels = extract_mnist_labels('train-labels-idx1-ubyte.gz', 60000)
test_images = extract_mnist_images('t10k-images-idx3-ubyte.gz', 10000)
test_labels = extract_mnist_labels('t10k-labels-idx1-ubyte.gz', 10000)

# 转换为PyTorch的Tensor格式
train_images = torch.from_numpy(train_images)
train_labels = torch.from_numpy(train_labels)
val_images = train_images[55000:]
val_labels = train_labels[55000:]
test_images = torch.from_numpy(test_images)
test_labels = torch.from_numpy(test_labels)


# 创建自定义Dataset实例
train_dataset = CustomDataset(train_images, train_labels, transform=transforms.ToTensor())
val_dataset = CustomDataset(val_images, val_labels, transform=transforms.ToTensor())
test_dataset = CustomDataset(test_images, test_labels, transform=transforms.ToTensor())

# 创建数据加载器
batch_size = 64
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 获取训练和验证数据
train_data, train_labels_n = flatten_data(train_loader)
test_data, test_labels = flatten_data(test_loader)

# 归一化KNN和SVM的数据
scaler = StandardScaler()
train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)

# 在不同的k值下测试KNN
k_values = [2, 4, 6, 8, 10, 12]
knn_train_test(train_data_normalized, train_labels_n, test_data_normalized, test_labels, k_values)


# 测试SVM
svm_accuracy = svm_train_test(train_data_normalized, train_labels_n, test_data_normalized, test_labels)
print(f"SVM Accuracy: {svm_accuracy}")

# 测试CNN
# 划分除验证集的部分
train_dataset = CustomDataset(train_images[:55000], train_labels[:55000], transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

cnn_accuracy = cnn_train_test(train_loader, val_loader, test_loader, 10)
print(f"CNN Accuracy: {cnn_accuracy}")
