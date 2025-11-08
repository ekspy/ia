# Cheatsheet para Aplicações em Python com Base nos Materiais

# === Manipulação de Dados com Pandas ===
import pandas as pd

# Criar e manipular DataFrame
data = {'Nome': ['Ana', 'Bruno', None], 'Idade': [23, None, 30], 'Salário': [4000, 5000, 3500]}
df = pd.DataFrame(data)

# Tratar valores ausentes
df['Idade'] = df['Idade'].fillna(df['Idade'].mean())  # Preenche com média
df.dropna(subset=['Nome'], inplace=True)  # Remove linhas com valores ausentes em 'Nome'

# Remover duplicatas
df = df.drop_duplicates()

# Padronizar texto
df['Nome'] = df['Nome'].str.lower()

# Criar nova coluna e filtrar
df['Salário Anual'] = df['Salário'] * 12
df_filtrado = df[df['Idade'] > 25]

# === Estatística e Probabilidade com NumPy e SciPy ===
import numpy as np
from scipy.stats import norm, binom, poisson

# Dados de exemplo para distribuição normal (altura de pessoas)
media = 70  # Média em pontos de teste
desvio = 10  # Desvio padrão
x = 85  # Valor observado (nota de um aluno)
z_score = (x - media) / desvio  # Z-score: (x - μ) / σ
prob_menor_88 = norm.cdf(0.88, loc=media, scale=desvio)  # Probabilidade de x < 0.88

# Dados para distribuição binomial (jogar moeda 10 vezes)
n = 10  # Número de tentativas
k = 3   # Número de sucessos
p = 0.5  # Probabilidade de sucesso (cara)
prob_exatos_3 = binom.pmf(k, n, p)  # P(X = 3) = (n choose k) * p^k * (1-p)^(n-k)

# Dados para distribuição Poisson (erros por hora)
mu = 2  # Média de eventos
k_poisson = 3  # Número de eventos
prob_3_erros = poisson.pmf(k_poisson, mu)  # P(X = 3)

print(f"Z-score: {z_score}, Prob < 0.88: {prob_menor_88:.4f}")
print(f"Probabilidade de 3 'caras': {prob_exatos_3:.4f}")
print(f"Probabilidade de 3 erros: {prob_3_erros:.4f}")

# === Visualização com Matplotlib e Seaborn ===
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de linha (tendência de notas)
x = np.linspace(0, 100, 100)
y = norm.pdf(x, loc=70, scale=10)  # Curva normal
plt.plot(x, y, label='Distribuição Normal')
plt.title('Distribuição de Notas')
plt.legend()
plt.show()

# Histograma com Seaborn
sns.histplot(df['Salário'], bins=5, kde=True)
plt.title('Distribuição de Salários')
plt.show()

# Heatmap de correlação
corr = df[['Idade', 'Salário', 'Salário Anual']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlação entre Variáveis')
plt.show()

# === Regressão Linear ===
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados de exemplo (tempo de estudo vs. nota)
X = df[['Idade']]  # Variável independente
y = df['Salário']   # Variável dependente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE da Regressão Linear: {mse:.2f}")

# === Regressão Logística ===
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Simulação de classificação (ex.: aprovação)
df['Aprovado'] = df['Salário'] > 4000  # Exemplo binário
X = df[['Idade']]
y = df['Aprovado']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_log = LogisticRegression()
modelo_log.fit(X_train, y_train)
y_pred_log = modelo_log.predict(X_test)
print(classification_report(y_test, y_pred_log))

# === Árvore de Decisão ===
from sklearn.tree import DecisionTreeClassifier, plot_tree

data_arvore = {'Peso': [150, 170], 'Textura': [1, 0], 'Fruta': ['maçã', 'laranja']}
df_arvore = pd.DataFrame(data_arvore)
X_arvore = df_arvore[['Peso', 'Textura']]
y_arvore = df_arvore['Fruta']

X_train_arvore, X_test_arvore, y_train_arvore, y_test_arvore = train_test_split(X_arvore, y_arvore, test_size=0.2, random_state=42)
modelo_arvore = DecisionTreeClassifier(max_depth=2, random_state=42)
modelo_arvore.fit(X_train_arvore, y_train_arvore)
plot_tree(modelo_arvore, feature_names=['Peso', 'Textura'], class_names=['maçã', 'laranja'], filled=True)
plt.show()

# === K-Means (Clusterização) ===
from sklearn.cluster import KMeans

X_kmeans = df[['Idade', 'Salário']]
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_kmeans)
print("Centróides:", kmeans.cluster_centers_)

# Método do Cotovelo
inercia = [KMeans(n_clusters=k, random_state=42).fit(X_kmeans).inertia_ for k in range(1, 6)]
plt.plot(range(1, 6), inercia, marker='o')
plt.title('Método do Cotovelo')
plt.xlabel('k')
plt.ylabel('Inércia')
plt.show()

# === Pipelines ===
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Pipeline com pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Idade', 'Salário']),
        ('cat', OneHotEncoder(), ['Nome'])
    ])
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
y_pred_pipeline = pipeline.predict(X_test)
print(classification_report(y_test, y_pred_pipeline))

# === Busca (BFS, DFS, UCS, Gulosa, A*) ===
grafo = {'A': ['B', 'C'], 'B': ['D'], 'C': ['D'], 'D': []}
distancias = {'A': 4, 'B': 2, 'C': 3, 'D': 0}

# BFS
from collections import deque
def bfs(grafo, inicio, objetivo):
    fila = deque([(inicio, [inicio])])
    visitados = set()
    while fila:
        no, caminho = fila.popleft()
        if no not in visitados:
            visitados.add(no)
            if no == objetivo:
                return caminho
            for vizinho in grafo.get(no, []):
                if vizinho not in visitados:
                    fila.append((vizinho, caminho + [vizinho]))
    return None

# DFS
def dfs(grafo, inicio, objetivo):
    pilha = [(inicio, [inicio])]
    visitados = set()
    while pilha:
        no, caminho = pilha.pop()
        if no not in visitados:
            visitados.add(no)
            if no == objetivo:
                return caminho
            for vizinho in grafo.get(no, []):
                if vizinho not in visitados:
                    pilha.append((vizinho, caminho + [vizinho]))
    return None

# UCS
import heapq
def ucs(grafo, custos, inicio, objetivo):
    fila = [(0, inicio, [inicio])]
    visitados = {}
    while fila:
        custo, no, caminho = heapq.heappop(fila)
        if no not in visitados or visitados[no] > custo:
            visitados[no] = custo
            if no == objetivo:
                return caminho
            for vizinho in grafo.get(no, []):
                novo_custo = custo + custos.get(vizinho, 0)
                heapq.heappush(fila, (novo_custo, vizinho, caminho + [vizinho]))
    return None

# Busca Gulosa
def busca_gulosa(grafo, distancias, inicio, objetivo):
    caminho = [inicio]
    atual = inicio
    while atual != objetivo:
        vizinhos = grafo[atual]
        atual = min(vizinhos, key=lambda x: distancias[x])
        caminho.append(atual)
    return caminho

# A*
def a_estrela(grafo, distancias, inicio, objetivo):
    fila = [(distancias[inicio], 0, inicio, [inicio])]
    custo_real = {cidade: float('inf') for cidade in grafo}
    custo_real[inicio] = 0
    while fila:
        _, g, atual, caminho = heapq.heappop(fila)
        if atual == objetivo:
            return caminho
        for vizinho in grafo[atual]:
            novo_g = g + 1  # Custo unitário simples
            if novo_g < custo_real[vizinho]:
                custo_real[vizinho] = novo_g
                f = novo_g + distancias[vizinho]
                heapq.heappush(fila, (f, novo_g, vizinho, caminho + [vizinho]))
    return None

print("BFS:", bfs(grafo, 'A', 'D'))
print("DFS:", dfs(grafo, 'A', 'D'))
print("UCS:", ucs(grafo, distancias, 'A', 'D'))
print("Gulosa:", busca_gulosa(grafo, distancias, 'A', 'D'))
print("A*:", a_estrela(grafo, distancias, 'A', 'D'))
