🔍 Comparativo de Algoritmos de Machine Learning - Previsão de Corridas Uber
🎯 Objetivo
Comparar o desempenho de três algoritmos de classificação diferentes na previsão de corridas completadas da Uber, analisando suas características e aplicabilidades.

📊 Sobre o Projeto
Este projeto realiza uma análise comparativa entre Regressão Logística, K-Nearest Neighbors (KNN) e Árvores de Decisão para entender qual algoritmo se adapta melhor aos padrões dos dados da Uber, considerando diferentes cenários e complexidades dos dados.

🧠 Algoritmos Comparados
1. Regressão Logística
```
LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
Melhor para: Relações predominantemente lineares
```
Vantagens: Interpretabilidade, probabilidades bem calibradas

Limitações: Assume linearidade entre features e log-odds

2. K-Nearest Neighbors (KNN)
```
KNeighborsClassifier(n_neighbors=5, weights='uniform')
Melhor para: Padrões locais complexos
```
Vantagens: Não assume forma funcional específica, simples

Limitações: Computacionalmente intensivo, sensível a escala

3. Árvore de Decisão
```
DecisionTreeClassifier(max_depth=5, random_state=42)
```
Melhor para: Capturar regras específicas e não-lineares

Vantagens: Interpretável, lida bem com features categóricas

Limitações: Tendência a overfitting, instável

📈 Features Utilizadas
Avg VTAT - Vehicle Turnaround Average Time

Avg CTAT - Customer Turnaround Average Time

🛠️ Implementação
Pipeline de Machine Learning
```
# Pipeline padrão para todos os algoritmos
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', algoritmo_escolhido)
])
```
Métricas de Avaliação
python
```
# Métricas calculadas para cada algoritmo
metrics = {
    'Accuracy': accuracy_score,
    'Precision': precision_score,
    'Recall': recall_score,
    'F1-Score': f1_score,
    'ROC AUC': roc_auc_score
}
```
📊 Resultados e Análise
Fronteira de Decisão
A visualização da fronteira de decisão permite comparar como cada algoritmo separa as classes:

```
# Visualização da fronteira de decisão
plt.contourf(xx, yy, zz, levels=25, cmap='RdBu', alpha=0.6)
plt.contour(xx, yy, zz, levels=[0.5], colors=['k'], linewidths=2)
```
Interpretação das fronteiras:

Linha preta (0.5): Limite de decisão (50% probabilidade)

Região azul: Probabilidade baixa de "completado"

Região vermelha: Probabilidade alta de "completado"

🎯 Cenários de Aplicação
✅ Regressão Logística
Dados com relações lineares claras

Quando interpretabilidade é crucial

Dataset com muitas features

✅ KNN
Padrões locais complexos

Datasets pequenos a médios

Quando a métrica de distância faz sentido

✅ Árvore de Decisão
Regras de negócio específicas

Features categóricas

Quando precisa de alta interpretabilidade

📋 Estrutura do Projeto
text
comparativo_algoritmos/
│
├── dados_uber.csv
├── comparativo_regressao_logistica.py
├── comparativo_knn.py
├── comparativo_arvore_decisao.py
├── analise_comparativa_final.py
└── README.md
🔧 Como Executar
bash
# Instalar dependências
pip install pandas numpy matplotlib scikit-learn

# Executar análise comparativa
python analise_comparativa_final.py
📊 Métricas de Comparação
Algoritmo	Accuracy	Precision	Recall	F1-Score	ROC AUC
Regressão Logística					
KNN					
Árvore de Decisão					
🎓 Conclusões Aprendidas
Não existe algoritmo universalmente melhor

A escolha depende da natureza dos dados

A visualização ajuda a entender o comportamento dos modelos

O balanceamento entre bias e variação é crucial

🚀 Próximos Passos
Implementar Ensemble Methods (Random Forest, Gradient Boosting)

Adicionar mais features ao modelo

Realizar tuning de hiperparâmetros

Implementar cross-validation

Desenvolver dashboard interativo
