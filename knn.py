import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# adding colomn names 
column_names = ["Class", "Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash",
                "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols",
                "Proanthocyanins", "Color Intensity", "Hue", "OD280/OD315", "Proline"]

df = pd.read_csv(url, names=column_names)
df.head()

scaler = MinMaxScaler()
df_scaled= df.copy()  #veriyi kaybetmmemek için kopyala
df_scaled.iloc[:, 1:] = pd.DataFrame(scaler.fit_transform(df.iloc[:, 1:]), 
                                     columns=df.columns[1:]).astype(float)
X = df_scaled.drop(columns=['Class'])
y = df_scaled['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=11, test_size=0.2)

class KNN:
    def __init__(self, k=3, metric = 'euclidean'):
        self.k = k 
        self.metric = metric

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def predict(self, X_test):
        return np.array([self._predict(x)[0] for x in np.array(X_test)]) 

    def _predict(self, x):
        distance = self.choose_dist_type(x)
        k_indices = np.argsort(distance)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        label = Counter(k_nearest_labels)
        most_common_label, count = label.most_common(1)[0]

        confidence = count / self.k
        return most_common_label, confidence  # Tuple döndü

    def choose_dist_type(self,x):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))  
        elif self.metric == 'manhattan':
            return np.sum(np.abs(self.X_train - x), axis=1) 
        elif self.metric == 'cosine':
            dot_product = np.sum(self.X_train * x, axis=1)
            norms = np.linalg.norm(self.X_train, axis=1) * np.linalg.norm(x)
            return 1 - (dot_product / norms)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred) * 100


k=3
knn3 = KNN(k=3, metric = 'euclidean') 
knn3.fit(X_train, y_train)  
y_pred = knn3.predict(X_test)  
acc = knn3.accuracy(y_test, y_pred)  #calcutae accuracy
accuracies.append(acc)  
print(f"K={k}, Accuracy={acc:.2f}%") 

print(f"total: {len(X)}")
print(f"traininf: {len(X_train)}")
print(f"test: {len(X_test)}")  #number of test data is 36 so we will see 36 test data in conf matrix
# for different k values, metic is euclidean
k_values = range(1,50)  
accuracies = []

for k in k_values:
    knn = KNN(k=k, metric='euclidean')  # Model
    knn.fit(X_train, y_train)  
    y_pred = knn.predict(X_test)  
    acc = knn.accuracy(y_test, y_pred)  #calcutae accuracy
    accuracies.append(acc)  
    print(f"K={k}, Accuracy={acc:.2f}%")  

#ploting
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b', label="Accuracy")
plt.xlabel("K Values")
plt.ylabel("accuracy (%)")
plt.title("KNN Model accuracy graph")
plt.legend()
plt.grid()
plt.show()

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# for different k values, metic is MANHATTAN
k_values = range(1,50)  
accuracies = []

for k in k_values:
    knn = KNN(k=k, metric='manhattan')  # Model
    knn.fit(X_train, y_train)  
    y_pred = knn.predict(X_test)  
    acc = knn.accuracy(y_test, y_pred)  #calcutae accuracy
    accuracies.append(acc)  
    print(f"K={k}, Accuracy={acc:.2f}%")  

#ploting
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b', label="Accuracy")
plt.xlabel("K Values")
plt.ylabel("accuracy (%)")
plt.title("KNN Model accuracy graph")
plt.legend()
plt.grid()
plt.show()

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# for different k values, metic is cosine
k_values = range(1,50)  
accuracies = []

for k in k_values:
    knn = KNN(k=k, metric='cosine')  # Model
    knn.fit(X_train, y_train)  
    y_pred = knn.predict(X_test)  
    acc = knn.accuracy(y_test, y_pred)  #calcutae accuracy
    accuracies.append(acc)  
    print(f"K={k}, Accuracy={acc:.2f}%")  

#ploting
plt.plot(k_values, accuracies, marker='o', linestyle='dashed', color='b', label="Accuracy")
plt.xlabel("K Values")
plt.ylabel("accuracy (%)")
plt.title("KNN Model accuracy graph")
plt.legend()
plt.grid()
plt.show()

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1", "Class 2"], yticklabels=["Class 0", "Class 1", "Class 2"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()