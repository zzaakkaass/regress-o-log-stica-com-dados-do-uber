📊 Modelo de Classificação - Previsão de Corridas Uber
🎯 Objetivo
Implementar um modelo de Regressão Logística para prever se uma corrida do Uber será completada ou não, baseado em métricas de tempo de espera.

📋 Sobre o Projeto
Este projeto representa minha evolução no aprendizado de Machine Learning com Python, aplicando conceitos de classificação binária em um dataset real da Uber. O foco está em entender como variáveis relacionadas ao tempo de espera impactam na conclusão das corridas.

🛠️ Tecnologias Utilizadas
Python 3

Pandas - Manipulação de dados

Scikit-learn - Machine Learning

Matplotlib - Visualização

NumPy - Computação numérica

📈 Features Utilizadas
Avg VTAT - Tempo médio de espera do veículo

Avg CTAT - Tempo médio de espera do cliente

🔧 Implementação
Pré-processamento
python
# Tratamento de valores nulos e conversão de tipos
```
df[feat1] = pd.to_numeric(df[feat1], errors='coerce')
df[feat2] = pd.to_numeric(df[feat2], errors='coerce')
```
# Criação da variável target binária

```
df['target_completed'] = (df['Booking Status'].str.lower() == 'completed').astype(int)
Pipeline de Machine Learning
python
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced', 
                                  solver='liblinear', 
                                  random_state=42))
])
```
📊 Métricas de Avaliação
O modelo é avaliado usando múltiplas métricas:

Accuracy - Acurácia geral

Precision - Precisão nas previsões positivas

Recall - Sensibilidade do modelo

F1-Score - Média harmônica entre precision e recall

ROC AUC - Área sob a curva ROC

🎨 Visualização
O projeto inclui uma visualização da fronteira de decisão que mostra:

Linha preta (nível 0.5): Limite de decisão do modelo

Cores do fundo: Probabilidade de ser classe "completada"

Pontos azuis/vermelhos: Instâncias de treino e teste

📁 Estrutura do Código
Carregamento e limpeza dos dados

Engenharia de features

Divisão treino/teste estratificada

Treinamento do modelo com pipeline

Avaliação com múltiplas métricas

Visualização da fronteira de decisão

🎓 Conceitos Aplicados
Classificação binária

Balanceamento de classes

Padronização de features

Validação estratificada

Interpretação de probabilidades

Análise de fronteira de decisão

🔍 Insights do Modelo
A fronteira de decisão permite visualizar:

Como o modelo separa as classes baseado nas features

Regiões de alta/baixa confiança nas previsões

Potenciais melhorias com modelos mais complexos

🚀 Próximos Passos
Testar outros algoritmos (KNN, Árvores de Decisão)

Adicionar mais features ao modelo

Realizar feature engineering mais avançado

Implementar cross-validation

Desenvolver API para previsões em tempo real
