#KNN tenha melhor performance se houver padrões locais complexos

#Árvore de Decisão seja boa para capturar regras específicas (ex: certos tipos de veículo em certos locais)

#Regressão Logística funcione bem se as relações forem predominantemente lineares


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)

# 1) Carregar a base
df = pd.read_csv(r'D:\meu-feed-personalizado\dados uber\dados_uber.csv',
                 na_values=['null','NULL','NaN','nan','None',''])

# 2) Criar alvo binário (Completed vs. outros)
df['target_completed'] = (
    df['Booking Status'].astype(str).str.strip().str.lower() == 'completed'
).astype(int)

# 3) Escolher duas features numéricas
feat1, feat2 = 'Avg VTAT', 'Avg CTAT'
df[feat1] = pd.to_numeric(df[feat1], errors='coerce')
df[feat2] = pd.to_numeric(df[feat2], errors='coerce')

# 4) Filtrar linhas válidas (ambas as features não-nulas)
mask = df[feat1].notna() & df[feat2].notna()
X = df.loc[mask, [feat1, feat2]].copy()
y = df.loc[mask, 'target_completed'].copy()

# 5) Dividir em treino/teste (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# 6) Pipeline: padronização + regressão logística
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(class_weight='balanced',
                                  solver='liblinear',
                                  random_state=42))
])
pipe.fit(X_train, y_train)

# 7) Avaliação
y_pred  = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)[:, 1]

print('Accuracy :', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall   :', recall_score(y_test, y_pred))
print('F1-Score :', f1_score(y_test, y_pred))
print('ROC AUC  :', roc_auc_score(y_test, y_proba))
print('Confusion matrix:\\n', confusion_matrix(y_test, y_pred))

# 8) Fronteira de decisão (grid sobre as duas features)
scaler = pipe.named_steps['scaler']
clf    = pipe.named_steps['logreg']

x_min, x_max = X[feat1].min() - 1, X[feat1].max() + 1
y_min, y_max = X[feat2].min() - 1, X[feat2].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]

# Transformar e prever probabilidade (classe 1 = Completed)
grid_scaled = scaler.transform(grid)
zz = clf.predict_proba(grid_scaled)[:, 1].reshape(xx.shape)

plt.figure(figsize=(8,6))
cs = plt.contourf(xx, yy, zz, levels=25, cmap='RdBu', alpha=0.6)
plt.colorbar(cs, label='P(Completed)')

# Pontos de treino/teste
plt.scatter(X_train[feat1], X_train[feat2], c=y_train, cmap='bwr',
            s=18, edgecolor='k', alpha=0.8, label='Treino')
plt.scatter(X_test[feat1],  X_test[feat2],  c=y_test,  cmap='bwr',
            s=28, marker='x', label='Teste')

# Curva de decisão
cs2 = plt.contour(xx, yy, zz, levels=[0.5], colors=['k'], linewidths=2)
plt.clabel(cs2, fmt={'0.5': '0.5'}, inline=True)

plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title('Logistic Regression – Fronteira de decisão\\nTarget: Completed (1) vs Outros (0)')
plt.legend()
plt.tight_layout()
plt.show()



#1. o que significa a linha preta (nível 0.5)
#probabilidade prevista pelo modelo é 50% para a corrida ser completada
#esquerda/abaixo da linha: região onde o modelo tende a prever classe 0 (não completado).
#direita/acima da linha: região onde o modelo tende a prever classe 1 (completado).

# 2. cores do fundo
#fundo é um mapa de calor da probabilidade P(Completed):
#tons azuis/vermelhos (dependendo da paleta) indicam regiões com probabilidade baixa ou alta.
#quanto mais intenso, mais confiante o modelo está.

#5. Como usar essa interpretação
#Se os pontos das duas classes estão bem separados por essa linha → modelo simples funciona bem.
#Se há muita mistura (pontos de classes diferentes em ambos os lados) → talvez precise de um modelo mais flexível (Árvore, KNN).