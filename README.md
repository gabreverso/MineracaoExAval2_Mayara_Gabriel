# Instruções para Executar os Códigos

Este repositório contém códigos em Python para treinar modelos de classificação e fazer previsões com base em conjuntos de dados. 

## Requisitos

Certifique-se de ter o Python instalado em sua máquina. Você pode baixá-lo em [python.org](https://www.python.org/downloads/).

Além disso, instale as dependências necessárias:

```bash
pip install pandas numpy scikit-learn joblib
```

## Como Executar

Clone o repositório em sua máquina local:

```bash
git clone https://github.com/mayyaiko/MineracaoExAval2_Mayara_Gabriel.git
```

Execute o código principal para treinar os modelos e fazer previsões:

```bash
python mineracao.py
```

Este comando executará o código principal, que treinará os modelos de classificação com os conjuntos de dados fornecidos e fará previsões com base nos conjuntos de teste. Os resultados das previsões serão exibidos no terminal.

Para visualizar os resultados salvos nos arquivos .joblib, você pode executar o arquivo joblibReader.py:

```bash
python joblibReader.py
```

Isso carregará os modelos salvos e fará previsões com eles, exibindo os resultados no terminal.

