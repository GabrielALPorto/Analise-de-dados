import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Obtendo os dados do sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# ":" pega todas as linhas no eixo x. ":-1" pega todas as colunas, exceto a última, pois Python é exclusivo no limite direito
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Divide os dados entre teste e treino. 20% para teste e o restante para treino.
# Tambem poderia ser 70% para treino, 10% para validação e 20% para teste
# A validacao e utilizada para comparação de diferentes modelos e/ou hiperparametros
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Usa a distribuicao normal, transforma a media dos dados padronizados em zero e o desvio padroa em 1
# Garante que os dados estejam todos em uma escala comum
# Simplifica o reconhecimento dos dados e evita problemas como Bias (enviesamento)
# Previne falhas de underfitting e overfitting
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 100 arvores de decisao
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Obtem a acuracia
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Plota a matriz de confusao
confusion_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,
            xticklabels=iris.target_names, yticklabels=iris.target_names)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Mostra as features mais importantes para as predicoes realizadas
feature_importances = classifier.feature_importances_

# Plota o grafico das predicoes
plt.barh(iris.feature_names, feature_importances)
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Classifier')
plt.show()