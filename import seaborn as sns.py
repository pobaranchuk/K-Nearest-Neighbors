import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt



breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)

print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

randomness = []
max_accuracy = []
for i in range (50, 151):
    
    
    training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = i)
    # print(len(training_data), len(training_labels))

    
    k_list = []
    accuracies = []
    
    for k in range (1, 101):
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(training_data, training_labels)
        k_list.append(k)
        accuracies.append(classifier.score(validation_data, validation_labels))

    randomness.append(i)
    max_accuracy.append(max(accuracies))
    
plt.plot(randomness, max_accuracy)
plt.xlabel("Random_State")
plt.ylabel("Max Validation Accuracy")
plt.title("Breast Cancer Classifier Accuracy")
plt.show()