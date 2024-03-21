from joblib import load
from mineracao import X_test_2c, X_test_3c

# Carregar os modelos salvos
decision_tree_2c = load('Decision Tree_2c.joblib')  # Carrega o modelo Decision Tree treinado com o dataset 2C
naive_bayes_2c = load('Naive Bayes_2c.joblib')  # Carrega o modelo Naive Bayes treinado com o dataset 2C
svm_2c = load('SVM_2c.joblib')  # Carrega o modelo SVM treinado com o dataset 2C

decision_tree_3c = load('Decision Tree_3c.joblib')  # Carrega o modelo Decision Tree treinado com o dataset 3C
naive_bayes_3c = load('Naive Bayes_3c.joblib')  # Carrega o modelo Naive Bayes treinado com o dataset 3C
svm_3c = load('SVM_3c.joblib')  # Carrega o modelo SVM treinado com o dataset 3C

# Fazer previsões com os modelos
y_pred_decision_tree_2c = decision_tree_2c.predict(X_test_2c)  # Faz previsões com o modelo Decision Tree treinado com o dataset 2C
y_pred_naive_bayes_2c = naive_bayes_2c.predict(X_test_2c)  # Faz previsões com o modelo Naive Bayes treinado com o dataset 2C
y_pred_svm_2c = svm_2c.predict(X_test_2c)  # Faz previsões com o modelo SVM treinado com o dataset 2C

y_pred_decision_tree_3c = decision_tree_3c.predict(X_test_3c)  # Faz previsões com o modelo Decision Tree treinado com o dataset 3C
y_pred_naive_bayes_3c = naive_bayes_3c.predict(X_test_3c)  # Faz previsões com o modelo Naive Bayes treinado com o dataset 3C
y_pred_svm_3c = svm_3c.predict(X_test_3c)  # Faz previsões com o modelo SVM treinado com o dataset 3C

# Imprimir as previsões
print("Decision Tree 2C:", y_pred_decision_tree_2c)  # Imprime as previsões feitas com o modelo Decision Tree treinado com o dataset 2C
print("Naive Bayes 2C:", y_pred_naive_bayes_2c)  # Imprime as previsões feitas com o modelo Naive Bayes treinado com o dataset 2C
print("SVM 2C:", y_pred_svm_2c)  # Imprime as previsões feitas com o modelo SVM treinado com o dataset 2C

print("Decision Tree 3C:", y_pred_decision_tree_3c)  # Imprime as previsões feitas com o modelo Decision Tree treinado com o dataset 3C
print("Naive Bayes 3C:", y_pred_naive_bayes_3c)  # Imprime as previsões feitas com o modelo Naive Bayes treinado com o dataset 3C
print("SVM 3C:", y_pred_svm_3c)  # Imprime as previsões feitas com o modelo SVM treinado com o dataset 3C
