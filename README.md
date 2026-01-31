# Equipe8_Ligia

# Sleep Quality MVP — Explicações Acionáveis

Este projeto é um **MVP (Prova de Conceito)** que prevê a qualidade do sono de uma pessoa
e gera **sugestões acionáveis e realistas** para melhorar esse resultado.

Ao invés de apenas classificar se o sono é bom ou ruim, o sistema responde:
> “O que a pessoa pode mudar, de forma prática, para melhorar sua qualidade de sono?”

---

##  Objetivo do MVP

Dado o perfil de uma pessoa (idade, nível de estresse, atividade física, passos diários, IMC etc.), o sistema:

- Prediz a qualidade do sono (**Boa** ou **Ruim**)
- Explica o resultado do modelo
- Sugere o **menor conjunto de mudanças possíveis** para melhorar o resultado
- Prioriza ações **realistas e executáveis** (counterfactual acionável)



---

## Dataset

**Sleep Health and Lifestyle Dataset (Kaggle)**

O dataset contém informações como:
- Idade, gênero e ocupação  
- Nível de atividade física e passos diários  
- Nível de estresse  
- IMC, pressão arterial e frequência cardíaca  
- Duração e qualidade do sono  
- Distúrbios do sono  

### Variável Target
A variável **Quality of Sleep** é transformada em binária:
- **Boa** → qualidade do sono ≥ 7  
- **Ruim** → qualidade do sono < 7  

---

##  Modelagem

- Pipeline de pré-processamento:
  - One-hot encoding para variáveis categóricas
  - Normalização de variáveis numéricas
- Separação treino/teste com seed fixa
- Modelos treinados:
  - Baseline: Regressão Logística
  - Modelo melhor: Random Forest
- Métricas de avaliação:
  - F1-score
  - ROC-AUC
  - Matriz de confusão

Temos na pasta "artifact" os pesos dos modelos

---

## Demo

O projeto inclui uma demo simples (Streamlit) que mostra o fluxo completo:
1. Entrada dos dados do usuário
2. Previsão da qualidade do sono
3. Geração de plano acionável com custo estimado

---

##  Como executar o projeto

```bash
pip install -r requirements.txt
streamlit run app/app.py
