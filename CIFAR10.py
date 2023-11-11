import torch
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
from torch.utils.data import Dataset

from CNN_CIFAR10 import cnn_train_test
from SVM import svm_train_test
from processing import CustomDataset, flatten_data, CIFAR10Loader

data_transform = transforms.Compose([
    transforms.ToPILImage(),  # 将Tensor转换为PIL Image
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转，参数表示旋转角度范围
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),  # 随机裁剪和缩放
    transforms.ToTensor(),  # 将PIL Image转换为Tensor
])
# 创建数据加载器
cifar_loader = CIFAR10Loader()

# 转换为PyTorch的Tensor格式
train_images = torch.from_numpy(cifar_loader.train_data).float()
train_labels = torch.from_numpy(cifar_loader.train_labels).long()
val_images = torch.from_numpy(cifar_loader.val_data).float()
val_labels = torch.from_numpy(cifar_loader.val_labels).long()
test_images = torch.from_numpy(cifar_loader.test_data).float()
test_labels = torch.from_numpy(cifar_loader.test_labels).long()

# 创建自定义Dataset实例
train_dataset = CustomDataset(train_images, train_labels, transform=data_transform)
val_dataset = CustomDataset(val_images, val_labels, transform=data_transform)
test_dataset = CustomDataset(test_images, test_labels, transform=data_transform)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 获取训练和验证数据
train_data, train_labels_n = flatten_data(train_loader)
test_data, test_labels = flatten_data(test_loader)

# 归一化KNN和SVM的数据
scaler = StandardScaler()
train_data_normalized = scaler.fit_transform(train_data)
test_data_normalized = scaler.transform(test_data)

# 测试SVM
svm_accuracy = svm_train_test(train_data_normalized, train_labels_n, test_data_normalized, test_labels)
print(f"SVM Accuracy: {svm_accuracy}")

# 测试CNN
# 划分除验证集的部分
train_dataset = CustomDataset(train_images, train_labels, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

cnn_accuracy = cnn_train_test(train_loader, val_loader, test_loader, 10,0.005)
print(f"CNN Accuracy: {cnn_accuracy}")
