import pandas as pd

from sklearn.metrics import confusion_matrix
from joblib import load
from mineracao import models
from joblibReader import X_test_2c
from joblibReader import X_test_3c
from mineracao import y_test_2c
from mineracao import y_test_3c

X_test_2c.rename(columns={'pelvic_tilt numeric': 'pelvic_tilt'}, inplace=True) # Renomeia a coluna 'pelvic_tilt numeric' para 'pelvic_tilt'

# Carregar os modelos salvos
decision_tree_2c = load('Decision Tree_2c.joblib')
naive_bayes_2c = load('Naive Bayes_2c.joblib')
svm_2c = load('SVM_2c.joblib')

decision_tree_3c = load('Decision Tree_3c.joblib')
naive_bayes_3c = load('Naive Bayes_3c.joblib')
svm_3c = load('SVM_3c.joblib')


# Calcula e imprime a matriz de confusão para cada modelo e dataset
for name, model in models.items():
    # Faz previsões para o dataset 2C
    y_pred_2c = model.predict(X_test_2c)
    cm_2c = confusion_matrix(y_test_2c, y_pred_2c)
    print(f'{name} - Dataset 2C - Confusion Matrix:\n{cm_2c}')
    
    # Faz previsões para o dataset 3C
    y_pred_3c = model.predict(X_test_3c)
    cm_3c = confusion_matrix(y_test_3c, y_pred_3c)
    print(f'{name} - Dataset 3C - Confusion Matrix:\n{cm_3c}')
