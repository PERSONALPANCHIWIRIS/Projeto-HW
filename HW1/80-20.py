import pandas as panda #Manipulação de dados
import numpy as np #Cálculos numéricos/arrays
import matplotlib.pyplot as plot #Graficos
import os #Para trabalhar com caminhos de arquivos
from sklearn.model_selection import train_test_split #Split
from sklearn.tree import DecisionTreeClassifier #Decision tree

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'hungarian_heart_diseases.csv')
data = panda.read_csv(csv_path)

#todas as linhas, todas as colunas exceto a ultima
x = data.iloc[:, : -1]
#todas as linhas, apenas a ultima coluna
y = data.iloc[:, -1]

#Divide a data e, dois grupos. 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=1
)

leaf_capacity = [1, 3, 5, 10, 25, 50, 100]
train_accuracy = []
test_accuracy = []

#Para cada valor de leaf_capacity, cria uma árvore de decisão
#Depois calcula a accuracy de treino e de teste
for leaf in leaf_capacity:
    tree = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=1)
    tree.fit(X_train, y_train)
    train_accuracy.append(tree.score(X_train, y_train))
    test_accuracy.append(tree.score(X_test, y_test))

plot.figure(figsize=(12,8))
#Cria os gráficos
plot.plot(leaf_capacity, train_accuracy, marker='o', label="Train Accuracy")
#Circulos para treino
plot.plot(leaf_capacity, test_accuracy, marker='s', label="Test Accuracy")
#Quadrados para teste
plot.xlabel("Min Samples per Leaf")
plot.ylabel("Accuracy")
plot.title("Decision Tree Accuracy by Minimum Samples per Leaf")
plot.legend()
plot.grid(True)
plot.savefig('80-20.png', dpi=300, bbox_inches='tight')
plot.close()
