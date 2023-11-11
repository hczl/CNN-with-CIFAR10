import os
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize


def svm_train_test(train_data, train_labels, test_data, test_labels):
    # 将标签进行二进制编码
    test_labels_bin = label_binarize(test_labels, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    svm_classifier = svm.SVC(C=100.0, kernel='rbf', gamma=0.03, verbose=True, probability=True)
    print('模型信息')
    print(svm_classifier.get_params())
    t1 = time.time()
    # 训练
    svm_classifier.fit(train_data, train_labels)
    t2 = time.time()
    SVMfit = float(t2 - t1)
    print("训练时间: {} seconds".format(SVMfit))
    # 评估
    svm_predictions = svm_classifier.predict(test_data)
    print('模型测试')

    # 混淆矩阵
    print(confusion_matrix(test_labels, svm_predictions))
    # f1-score,precision,recall
    print(classification_report(test_labels, np.array(svm_predictions)))
    # 计算SVM模型的ROC曲线和AUC
    fpr_svm = dict()
    tpr_svm = dict()
    roc_auc_svm = dict()
    for i in range(10):
        fpr_svm[i], tpr_svm[i], _ = roc_curve(test_labels_bin[:, i], svm_classifier.predict_proba(test_data)[:, i])
        roc_auc_svm[i] = auc(fpr_svm[i], tpr_svm[i])

    # 绘制SVM模型的ROC曲线
    plt.figure()
    for i in range(10):
        plt.plot(fpr_svm[i], tpr_svm[i], label=f'SVM Class {i} (AUC = {roc_auc_svm[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC Curve for Each Class')
    plt.legend(loc="lower right")
    plt.show()

    accuracy = accuracy_score(test_labels, svm_predictions)
    return accuracy