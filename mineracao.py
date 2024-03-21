# ============= instalações no terminal =============
# =  pip install pandas numpy scikit-learn joblib   =
# ===================================================

# Importação das bibliotecas necessárias
import pandas as pd  # Para manipulação dos dados
import numpy as np  # Para operações numéricas
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.tree import DecisionTreeClassifier  # Para usar o modelo de Árvore de Decisão
from sklearn.naive_bayes import GaussianNB  # Para usar o modelo Naive Bayes
from sklearn.svm import SVC  # Para usar o modelo SVM
from sklearn.metrics import accuracy_score  # Para calcular a acurácia
from joblib import dump  # Para ler e salvar os modelos treinados


# Carregar os datasets
data_2c = pd.read_csv('column_2C_weka.csv')
data_3c = pd.read_csv('column_3C_weka.csv')

# Pré-processamento: remover instâncias com valores ausentes
data_2c.dropna(inplace=True)
data_3c.dropna(inplace=True)

# Separar os dados em features (X) e labels (y)
X_2c = data_2c.drop('class', axis=1)
y_2c = data_2c['class']
X_3c = data_3c.drop('class', axis=1)
y_3c = data_3c['class']

# Separar os dados em treino e teste
X_train_2c, X_test_2c, y_train_2c, y_test_2c = train_test_split(X_2c, y_2c, test_size=20, random_state=42)
X_train_3c, X_test_3c, y_train_3c, y_test_3c = train_test_split(X_3c, y_3c, test_size=20, random_state=42)

# Definir os modelos de classificação
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC()
}

# Para cada modelo, treinar, salvar o modelo, fazer predição e calcular a acurácia
for name, model in models.items():
    # Treinar o modelo e salvar
    model.fit(X_train_2c, y_train_2c)
    dump(model, f'{name}_2c.joblib')
    # Fazer predição e calcular a acurácia para o dataset 2C
    y_pred_2c = model.predict(X_test_2c)
    accuracy_2c = accuracy_score(y_test_2c, y_pred_2c)
    
    # Treinar o modelo e salvar
    model.fit(X_train_3c, y_train_3c)
    dump(model, f'{name}_3c.joblib')

    # Fazer predição e calcular a acurácia para o dataset 3C
    y_pred_3c = model.predict(X_test_3c)
    accuracy_3c = accuracy_score(y_test_3c, y_pred_3c)
    
    # Imprimir a acurácia para cada modelo e dataset
    print(f'{name} - Dataset 2C - Accuracy: {accuracy_2c}')
    print(f'{name} - Dataset 3C - Accuracy: {accuracy_3c}')