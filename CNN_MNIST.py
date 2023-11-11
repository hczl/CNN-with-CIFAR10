import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns

# 设置是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device : {device}")


class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.layer1 = torch.nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.layer2 = torch.nn.MaxPool2d(kernel_size=2)
        self.layer3 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.layer4 = torch.nn.MaxPool2d(kernel_size=2)
        self.layer5 = torch.nn.Linear(7 * 7 * 64, 1000)
        self.layer6 = torch.nn.Linear(1000, 10)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        x = torch.relu(self.layer3(x))
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.layer5(x))
        x = self.layer6(x)
        return x


# 训练和测试CNN模型
def cnn_train_test(train_loader, val_loader, test_loader, num_epochs=5, learning_rate=0.001):
    model = CNNModel().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    all_labels = []
    all_predictions = []
    matrix_labels = []
    matrix_predictions = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_correct = 0
        val_total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(outputs.data.cpu().numpy())

        val_accuracy = val_correct / val_total
        epoch_accuracies.append(val_accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy}")

    model.eval()
    test_correct = 0
    test_total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        matrix_labels.extend(labels.cpu().numpy())
        matrix_predictions.extend(predicted.cpu().numpy())

    test_accuracy = test_correct / test_total

    # 计算多类别问题的 ROC 曲线
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # 对每个类别计算 ROC 曲线和 AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])  # 传递 tpr[i] 参数

    # 绘制 ROC 曲线
    plt.figure()
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt.show()

    # 计算混淆矩阵
    conf_matrix = confusion_matrix(matrix_labels, matrix_predictions, labels=range(10))

    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

    # 可视化每个 epoch 的准确率
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_accuracies, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over Epochs')
    plt.show()

    return test_accuracy
