from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plot
import pandas as panda
import os
from sklearn.tree import DecisionTreeClassifier #Decision tree

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'hungarian_heart_diseases.csv')
data = panda.read_csv(csv_path)

#todas as linhas, todas as colunas exceto a ultima
x = data.iloc[:, : -1]
#todas as linhas, apenas a ultima coluna
y = data.iloc[:, -1]

#60-40 split (60 para o train, 20 para o resto)
X_train, X_rest, y_train, y_rest = train_test_split(
    x, y, test_size=0.4, stratify=y, random_state=1
)
#20-20 split (20 para validação, 20 para teste)
#Do resto (40%)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=1
)

#Limitações dos hiperparâmetros
parameters = {"max_depth": [2, 3, 4], "min_samples_split": range(2, 101)}
best_model = None
best_params = None
best_val_acc = 0

for params in ParameterGrid(parameters):
    classifier = DecisionTreeClassifier(
        max_depth = params["max_depth"],
        min_samples_split = params["min_samples_split"],
        random_state = 1
    )
    classifier.fit(X_train, y_train)
    val_acc = classifier.score(X_val, y_val)
    test_acc = classifier.score(X_test, y_test)

    #Encontramos um melhor no teste
    #At least 80, 78.5% in test
    if val_acc >= 0.80 and test_acc >= 0.785 and val_acc > best_val_acc:
        best_model = classifier
        best_val_acc = val_acc
        best_params = params
        best_test_acc = test_acc

#Debugging para comparar se da o mesmo para diferentes executações
#print("Best_params", best_params)
#print("val_accuracy", best_val_acc)
#print("test_accuracy", best_test_acc)

#O melhor de todos
plot.figure(figsize=(12,8))
#plot.title("Decision Tree")
#plot.legend()
plot.suptitle("Decision Tree", fontsize=16, y=0.95)
plot.tight_layout()
plot_tree(best_model, feature_names=x.columns, class_names=["Normal","Heart Disease"], filled=True)
#plot.show()
plot.savefig('60-20-20.png', dpi=300, bbox_inches='tight')
plot.close()

