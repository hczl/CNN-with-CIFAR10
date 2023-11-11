from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def knn_train_test(train_data, train_labels, test_data, test_labels, k_values):
    k_accuracy = []
    for k in k_values:
        knn_classifier = KNeighborsClassifier(n_neighbors=k)
        knn_classifier.fit(train_data, train_labels)
        knn_predictions = knn_classifier.predict(test_data)
        accuracy = accuracy_score(test_labels, knn_predictions)
        k_accuracy.append(accuracy)
        print(f"KNN (k={k}) Accuracy: {accuracy}")
    plt.figure()
    plt.plot(k_values, k_accuracy, marker='o')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy over k')
    plt.show()
