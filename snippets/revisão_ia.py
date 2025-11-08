"""
## **6. Regressão Linear & Regressão Logística**

Utilize o conjunto de dados MPG Dataset disponıvel em:

https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv

Realize as seguintes etapas:
- Treine uma Regressão Linear para prever o consumo de combustível (mpg) em função de horsepower and weight.
- Crie uma nova variável categórica binária chamada Eficiente, que vale 1 se mpg > 25 e 0 caso contrário:

      df[’Eficiente’] = (df[’mpg’] > 25).astype(int)

- Treine uma Regressão Logıstica para prever a variável Eficiente com base em horsepower e weight.
- Após o treinamento, use ambos os modelos para prever o consumo e a eficiência de um novo veıculo com:
horsepower = 90, weight = 2200
- Indique a métrica utilizada em cada modelo (ex: R2, acurácia).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")

df['Eficiente'] = (df['mpg'] > 25).astype(int)

df['horsepower'] = df['horsepower'].fillna(df['horsepower'].mean())

XLinear = df[['horsepower', 'weight']]
YLinear = df['mpg']

XLogistico = df[['horsepower', 'weight']]
YLogistico = df['Eficiente']

modeloLinear = LinearRegression()
modeloLinear.fit(XLinear, YLinear)

modeloLogistico = LogisticRegression()
modeloLogistico.fit(XLogistico, YLogistico)

dadoNovo = pd.DataFrame({'horsepower': [90], 'weight': [2200]})

YpredLinear = modeloLinear.predict(dadoNovo)
YpredLogistico = modeloLogistico.predict(dadoNovo)

print(f"Linear novo: ", YpredLinear)
print(f"Logistico novo: ", YpredLogistico)

print(f"R2 Linear: ", r2_score(YLinear, modeloLinear.predict(XLinear)))
print(f"R2 Logistico: ", accuracy_score(YLogistico, modeloLogistico.predict(XLogistico)))

"""## **7. K-means**

Utilize o conjunto de dados Iris Dataset disponıvel em:

https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv

Aplique o algoritmo K-means para segmentar as amostras com base nas variaveis numericas:
- Padronize os dados antes do agrupamento.
- Utilize o método do cotovelo ou silhueta para definir o número ideal de clusters.
- Mostre os clusters graficamente em 2D (exemplo: duas primeiras variáveis principais).
- Descreva as características principais de cada grupo.

"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

kf = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

X = kf[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.show()

optimal_k = 3

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=optimal_k, random_state=42))
])

pipeline.fit(X)
kf['cluster'] = pipeline.predict(X)

plt.scatter(kf['sepal_length'], kf['sepal_width'], c=kf['cluster'])
centroids = pipeline.named_steps['kmeans'].cluster_centers_
centroids_original_scale = pipeline.named_steps['scaler'].inverse_transform(centroids)
plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1],
            c='red', s=200, marker='X', label='Centróides')
plt.show()

print("\nCharacteristics of each cluster:")
print(kf.drop('species', axis=1).groupby('cluster').mean())

"""## **8. Árvore de Decisão**

Utilize o conjunto de dados Titanic Dataset disponıvel em:

https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Construa uma Arvore de Decisao  para prever a sobrevivencia dos passageiros com base nos atributos:

    Sex, Age, Pclass, Fare, SibSp, Parch, Embarked

Apresente:
- A árvore gerada (visualização gráfica);
- A acurácia do modelo;
- Uma breve discussão sobre interpretabilidade e limitações.
Após o treinamento, utilize o modelo para prever a sobrevivência de um novo passageiro com:

      Sexo = feminino, Idade = 25, Classe = 2, Tarifa = 15.0, Embarque = C, Irmaos/conjuges = 1, Pais/filhos

"""

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dt = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

dt['Age'] = dt['Age'].fillna(dt['Age'].mean())
dt['Embarked'] = dt['Embarked'].fillna(dt['Embarked'].mode()[0])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Age', 'Fare', 'SibSp', 'Parch']),
        ('cat', OneHotEncoder(), ['Sex', 'Pclass', 'Embarked'])
    ],
    remainder='passthrough'
)

X = dt[['Sex', 'Age', 'Pclass', 'Fare', 'SibSp', 'Parch', 'Embarked']]
Y = dt['Survived']

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42))
])

pipeline.fit(X, Y)

fitted_preprocessor = pipeline.named_steps['preprocessor']
num_feature_names = fitted_preprocessor.transformers_[0][2]
cat_feature_names = fitted_preprocessor.transformers_[1][1].get_feature_names_out(['Sex', 'Pclass', 'Embarked'])
feature_names_out = list(num_feature_names) + list(cat_feature_names)


plt.figure(figsize=(20,10))
plot_tree(pipeline.named_steps['classifier'], feature_names=feature_names_out, class_names=['Not Survived', 'Survived'], filled=True)
plt.show()

Y_pred = pipeline.predict(X)
accuracy = (Y_pred == Y).mean()
print(f"Accuracy of the Decision Tree Model: {accuracy:.4f}")

print("\nDiscussion:")
print("- Interpretability: Decision trees are generally easy to interpret, especially with limited depth. The plot shows clear rules (splits) at each node leading to a decision.")
print("- Limitations: Decision trees can be prone to overfitting, especially with greater depth. They can also be unstable, meaning small changes in the data can lead to a very different tree.")

new_passenger = pd.DataFrame({
    'Sex': ['female'],
    'Age': [25],
    'Pclass': [2],
    'Fare': [15.0],
    'SibSp': [1],
    'Parch': [1],
    'Embarked': ['C']
})

predicted_survival = pipeline.predict(new_passenger)
predicted_survival_proba = pipeline.predict_proba(new_passenger)[:, 1]

print(f"\nPrediction for the new passenger:")
print(f"- Predicted Survival (0=No, 1=Yes): {predicted_survival[0]}")
print(f"- Predicted Probability of Survival: {predicted_survival_proba[0]:.4f}")
