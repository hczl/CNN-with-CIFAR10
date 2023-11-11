import os
import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, label = self.images[index].reshape(3, 32, 32), int(self.labels[index])

        # 如果有transform，应用transform
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        else:
            # 如果没有transform，确保图像是torch.Tensor类型
            image = image.clone().detach().requires_grad_(True)

        return image, label

class CIFAR10Loader:
    def __init__(self, data_path='cifar-10-batches-py'):
        self.data_path = data_path
        self.label_names = self.load_label_names()
        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = self.load_data()

    def load_label_names(self):
        meta_data = self.unpickle(os.path.join(self.data_path, 'batches.meta'))
        return [label.decode('utf-8') for label in meta_data[b'label_names']]

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def load_data(self):
        # 加载训练数据
        train_data_batches = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        train_data = []
        train_labels = []

        for batch in train_data_batches:
            data_batch = self.unpickle(os.path.join(self.data_path, batch))
            train_data.append(data_batch[b'data'])
            train_labels += data_batch[b'labels']

        # 加载测试数据
        test_data_batch = self.unpickle(os.path.join(self.data_path, 'test_batch'))
        test_data = test_data_batch[b'data']
        test_labels = np.array(test_data_batch[b'labels'])

        # 合并训练数据
        train_data = np.vstack(train_data)
        train_labels = np.array(train_labels)

        # 划分训练和验证集
        train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels,
                                                                          test_size=0.1, random_state=11)

        return train_data, train_labels, val_data, val_labels, test_data, test_labels

    def get_label_name(self, label):
        return self.label_names[label]


# 转换数据为numpy数组
def flatten_data(loader):
    data = []
    labels = []
    for images, targets in loader:
        data.append(images.view(images.size(0), -1).detach().numpy())
        labels.append(targets.numpy())
    return np.vstack(data), np.concatenate(labels)
