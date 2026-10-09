
# Tech Challenge — Fase 2 | Pós-Tech FIAP — Data Analytics

## Classificação de risco de crédito com Machine Learning

Projeto desenvolvido como parte do Tech Challenge da Fase 2 da Pós-Tech em Data Analytics da FIAP.

O trabalho apresenta a construção de uma base analítica de crédito, a definição de uma variável-alvo, o tratamento dos dados, a comparação de algoritmos de classificação e a avaliação dos resultados, considerando o desbalanceamento das classes e a interpretação das variáveis explicativas.

---


## 1. Identificação

| Campo | Informação |
|---|---|
| Instituição | FIAP |
| Curso | Pós-Tech em Data Analytics |
| Atividade | Tech Challenge — Fase 2 |
| Turma | 2DTATBB |
| Grupo | 37 |
| Data de entrega | 10-10-2026 |

### Integrantes

| Nome completo | RM | E-mail |
|---|---|---|
| Eriscley Ferreira Mota | [INFORMAR RM] | Eriscley@gmail.com |
| Andre Luis Dias Pinto | [INFORMAR RM] | andrelsds@yahoo.com.br |
| Vitor Renato Michelucci Jose | rm377769 | vitor.michelucci@bb.com.br |
| Andre Rocha de Oliveira | [INFORMAR RM] | andrerochadeoliveira@gmail.com |
| Joao Victor Espindola Couto | [INFORMAR RM] | joao.couto@bb.com.br |

---

## 2. Links da entrega

| Item | Link |
|---|---|
| Repositório GitHub | https://github.com/andrerochadeoliveira/postech-tech-challenge-fase2-template |
| Vídeo executivo (até 5 minutos) | [INSERIR LINK] |
| Apresentação executiva | [INSERIR LINK] |

Os links devem estar acessíveis aos avaliadores e corresponder aos informados no documento de submissão.

---

## 3. O problema

A avaliação do risco de crédito é uma atividade relevante para instituições financeiras, pois contribui para identificar perfis associados a maior probabilidade de apresentar dificuldades no cumprimento de obrigações.

Neste projeto, investigamos a aplicação de algoritmos de Machine Learning para classificar registros de crédito com base em informações cadastrais e socioeconômicas.

O objetivo é comparar diferentes estratégias de classificação, avaliar sua capacidade de identificação da classe positiva e compreender quais características apresentam maior contribuição para as previsões.

O estudo possui finalidade acadêmica e experimental. Os resultados não devem ser interpretados como uma solução pronta para decisões automáticas de concessão de crédito.

### 3.1. Fonte dos dados

Foi utilizado o dataset público **Credit Card Approval Prediction**, disponibilizado na plataforma Kaggle.

**Fonte:** https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data

O conjunto original contém dois arquivos:

- `application_record.csv`: informações cadastrais e socioeconômicas dos clientes.
- `credit_record.csv`: histórico mensal de situação de crédito, utilizado para construção da variável-alvo.

Os arquivos são relacionados pelo identificador `ID`.

### 3.2. Construção da variável-alvo

A variável-alvo `target` foi construída no notebook `00_preparacao_dados.ipynb`, utilizando o histórico mensal de crédito.

Foram considerados clientes com registros completos em uma janela de 13 meses consecutivos, entre `MONTHS_BALANCE = -12` e `MONTHS_BALANCE = 0`.

A classificação original foi definida da seguinte forma:

- **target = 1:** pelo menos uma ocorrência de atraso igual ou superior a 60 dias, ou pelo menos duas ocorrências mensais de atraso igual ou superior a 30 dias.
- **target = 0:** registros que não atendem aos critérios anteriores.

Após a integração das bases e a seleção dos registros elegíveis, a base analítica apresentou:

| Classe | Quantidade | Proporção |
|---|---:|---:|
| target = 0 | 16.010 | 96,79% |
| target = 1 | 531 | 3,21% |
| **Total** | **16.541** | **100%** |

### 3.3. Consolidação de perfis cadastrais

Durante a análise exploratória e o pré-processamento, foram identificados registros com características cadastrais equivalentes, embora associados a identificadores distintos.

Para controlar esses registros, foi criado o atributo `grupo_perfil`, com base nas características explicativas, excluindo `ID` e `target`.

Foram identificados **6.923 perfis distintos**, dos quais **274 apresentaram classificações conflitantes da variável-alvo**.

Como decisão metodológica experimental, adotou-se a regra de atribuir `target = 1` a todos os registros de um perfil conflitante quando pelo menos um deles apresentava essa classificação.

Após essa consolidação, a distribuição passou a ser:

| Classe | Quantidade | Proporção |
|---|---:|---:|
| target = 0 | 15.233 | 92,09% |
| target = 1 | 1.308 | 7,91% |
| **Total** | **16.541** | **100%** |

**Ressalva metodológica:** essa consolidação redefine operacionalmente a variável-alvo. Os registros reclassificados não representam necessariamente novos eventos individuais de inadimplência observados. Portanto, o modelo final prevê a classe positiva consolidada por perfil cadastral.

Essa decisão constitui uma limitação do estudo e deve ser considerada na interpretação dos resultados.

### 3.4. Principais variáveis

| Variável | Descrição |
|---|---|
| `ID` | Identificador do registro |
| `AMT_INCOME_TOTAL` | Renda declarada |
| `CNT_CHILDREN` | Quantidade de filhos |
| `CNT_FAM_MEMBERS` | Quantidade de membros da família |
| `DAYS_BIRTH` | Idade representada em dias |
| `DAYS_EMPLOYED` | Tempo de emprego representado em dias |
| `NAME_INCOME_TYPE` | Categoria de fonte de renda |
| `NAME_EDUCATION_TYPE` | Escolaridade |
| `NAME_FAMILY_STATUS` | Estado civil |
| `NAME_HOUSING_TYPE` | Tipo de moradia |
| `OCCUPATION_TYPE` | Ocupação profissional |
| `FLAG_OWN_CAR` | Indicador de posse de automóvel |
| `FLAG_OWN_REALTY` | Indicador de posse de imóvel |
| `target` | Variável-alvo de classificação |
| `grupo_perfil` | Identificador de agrupamento de perfis cadastrais equivalentes |

Durante o pré-processamento também foram criados atributos derivados, como renda por membro familiar, relação entre tempo de emprego e idade e indicador simplificado de posse de bens.

A base final pré-processada contém **16.541 registros e 53 colunas**, sendo 50 variáveis preditoras, além de `ID`, `grupo_perfil` e `target`.

---

## 4. Metodologia e reprodução

### 4.1. Organização das etapas

O projeto está organizado em cinco notebooks:

| Ordem | Notebook | Finalidade |
|---|---|---|
| 00 | `00_preparacao_dados.ipynb` | Obtenção dos dados, análise dos identificadores e construção do target |
| 01 | `01_eda.ipynb` | Análise exploratória dos dados |
| 02 | `02_preprocessamento.ipynb` | Tratamento, transformação, engenharia de atributos e consolidação dos perfis |
| 03 | `03_modelagem.ipynb` | Treinamento, validação cruzada e comparação dos modelos |
| 04 | `04_avaliacao.ipynb` | Avaliação final, matriz de confusão, importância das variáveis e conclusões |

### 4.2. Obtenção dos dados

O notebook 00 utiliza a biblioteca `kagglehub` para obter os arquivos originais do Kaggle e disponibilizá-los em `data/raw/`.

Os arquivos esperados são:

- `application_record.csv`
- `credit_record.csv`

A execução gera `data/processed/dataset_base.csv`, utilizado nas etapas de análise exploratória e pré-processamento.

O notebook 02 produz `data/processed/dataset_preprocessed.csv`, utilizado nos notebooks 03 e 04.

### 4.3. Execução no Google Colab

Os notebooks foram desenvolvidos para execução no Google Colab, com configuração do caminho do projeto em:

`/content/postech-tech-challenge-fase2-template`

Para iniciar uma sessão limpa, pode-se clonar o repositório:

```python
!git clone https://github.com/andrerochadeoliveira/postech-tech-challenge-fase2-template.git
```

Em seguida, os notebooks devem ser executados na ordem lógica de 00 a 04.

Alguns notebooks possuem comandos que executam automaticamente etapas anteriores por meio de `jupyter nbconvert`.

Para reprodução, é necessário que as bibliotecas utilizadas estejam instaladas e que os comandos sejam executados no ambiente esperado.

### 4.4. Pré-processamento

As principais transformações realizadas foram:

- Tratamento de valores ausentes em `OCCUPATION_TYPE`, preservando a ausência como categoria específica.
- Conversão de variáveis categóricas em indicadores binários.
- Transformação das variáveis de idade e tempo de emprego.
- Remoção de variável constante.
- Criação de atributos derivados.
- Identificação de perfis cadastrais equivalentes.
- Consolidação experimental da variável-alvo nos perfis conflitantes.

A padronização das variáveis foi realizada na etapa de modelagem, dentro dos pipelines, para evitar que seus parâmetros fossem estimados utilizando dados de validação ou teste.

### 4.5. Separação dos dados

Foi utilizado `StratifiedGroupKFold`, com semente aleatória fixa (`random_state = 42`).

A separação inicial resultou em:

| Conjunto | Registros | Positivos |
|---|---:|---:|
| Treinamento | 13.255 | 997 |
| Teste | 3.286 | 311 |

O agrupamento por `grupo_perfil` impede que registros de um mesmo perfil cadastral sejam compartilhados entre treinamento e teste.

### 4.6. Algoritmos avaliados

Foram comparados dois classificadores:

- Regressão Logística.
- Random Forest.

Para cada algoritmo, foram avaliadas três estratégias:

1. Modelo baseline, sem tratamento específico do desbalanceamento.
2. Ponderação de classes (`class_weight="balanced"`).
3. Sobreamostragem da classe minoritária com SMOTE.

Ao todo, foram avaliadas **seis configurações**.

A comparação utilizou validação cruzada estratificada por grupos, com cinco folds, exclusivamente no conjunto de treinamento.

O **F1-score** foi utilizado como principal critério de seleção, acompanhado de Precision, Recall, ROC-AUC e Average Precision.

---

## 5. Resultados

### 5.1. Seleção do modelo

A **Regressão Logística com ponderação de classes** apresentou o maior F1-score médio na validação cruzada.

| Métrica | Média na validação cruzada |
|---|---:|
| F1-score | 0,1485 |
| Precision | 0,0882 |
| Recall | 0,4704 |
| ROC-AUC | 0,5339 |
| Average Precision | 0,1026 |

A Regressão Logística com SMOTE apresentou desempenho muito próximo, com F1 médio de aproximadamente 0,1481. Essa pequena diferença não permite concluir que a estratégia selecionada seja substancialmente superior.

### 5.2. Definição do ponto de corte

O threshold foi investigado utilizando previsões *out-of-fold* produzidas exclusivamente com o conjunto de treinamento.

O ponto de corte de **0,51** apresentou o maior F1 nessa etapa, aproximadamente **0,1515**, e foi mantido para a avaliação final.

Embora o threshold padrão de 0,50 tenha apresentado F1 superior no conjunto de teste, o valor selecionado não foi alterado, evitando utilizar o teste para otimização do modelo.

### 5.3. Avaliação final

O modelo selecionado foi avaliado em 3.286 registros de teste.

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,6595 |
| Precision | 0,1189 |
| Recall | 0,4051 |
| F1-score | 0,1838 |
| ROC-AUC | 0,5814 |
| Average Precision | 0,1556 |

### 5.4. Matriz de confusão

| | Previsto negativo | Previsto positivo |
|---|---:|---:|
| Real negativo | 2.041 | 934 |
| Real positivo | 185 | 126 |

O modelo identificou corretamente **126 dos 311 registros positivos**, mas também produziu **934 falsos positivos**.

Isso representa uma Precision de 11,89% e um Recall de 40,51%.

Os resultados evidenciam limitações relevantes para utilização operacional, principalmente pela elevada quantidade de alertas incorretos e pela proporção de positivos não identificados.

### 5.5. Importância das variáveis

A interpretação foi realizada por meio de duas abordagens:

- Análise dos coeficientes padronizados da Regressão Logística.
- Permutation Importance, utilizando Average Precision como métrica de referência.

Entre as variáveis com destaque nas análises estão:

- `Employed`
- `Ratio_life_employed`
- `NAME_EDUCATION_TYPE_Higher education`
- `Year_employed`
- `NAME_INCOME_TYPE_Pensioner`

Esses resultados sugerem associações entre as características profissionais, educacionais e as previsões do modelo.

As importâncias representam associações estatísticas ou contribuições preditivas, não relações causais.

---

## 6. Principais conclusões

1. **O desbalanceamento das classes é uma característica central do problema.** A classe positiva representa 7,91% da base consolidada, tornando insuficiente a avaliação baseada apenas em acurácia.

2. **A Regressão Logística ponderada apresentou o melhor F1 médio entre as configurações avaliadas**, embora seu desempenho tenha sido muito próximo ao da Regressão Logística com SMOTE.

3. **A separação por perfis cadastrais foi importante para a validação.** A estratégia impediu o compartilhamento de perfis equivalentes entre treinamento e teste.

4. **O desempenho final ainda é limitado para utilização prática.** A baixa Precision e o volume de falsos positivos indicam que o modelo não deve ser utilizado autonomamente em decisões de crédito.

5. **A interpretação identificou contribuições de características profissionais e educacionais**, mas os resultados não demonstram causalidade e devem ser analisados com cautela.

### 6.1. Limitações

As principais limitações identificadas foram:

- Capacidade discriminativa limitada.
- Elevada quantidade de falsos positivos.
- Desbalanceamento das classes.
- Ausência de informações comportamentais adicionais.
- Redefinição operacional da variável-alvo durante a consolidação dos perfis.
- Ausência de validação temporal e externa.
- Possível influência da correlação entre variáveis na interpretação das importâncias.

### 6.2. Próximos passos

Como possibilidades de aprimoramento, destacam-se:

- Avaliação de algoritmos adicionais, incluindo métodos de boosting.
- Inclusão de informações históricas e comportamentais.
- Investigação de estratégias alternativas de tratamento do desbalanceamento.
- Comparação entre a variável-alvo original e a consolidada.
- Avaliação da calibração das probabilidades.
- Definição de thresholds orientados por custos de negócio.
- Validação temporal e externa.

O trabalho demonstra a aplicação de um processo estruturado de Machine Learning, mas seus resultados possuem caráter experimental.

---

## 7. Estrutura do repositório

```text
postech-tech-challenge-fase2-template/
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 00_preparacao_dados.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_preprocessamento.ipynb
│   ├── 03_modelagem.ipynb
│   └── 04_avaliacao.ipynb
├── docs/
├── requirements.txt
└── README.md
```

Os diretórios `data/raw/` e `data/processed/` são utilizados para armazenamento local dos arquivos e das bases geradas.

Consulte também os documentos `ESTRUTURA.md` e `CHECKLIST.md` do repositório.

---

## 8. Tecnologias utilizadas

- Python
- Google Colab
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- imbalanced-learn
- kagglehub
- Git e GitHub

A reprodutibilidade depende da instalação de versões compatíveis dessas bibliotecas e da disponibilidade dos arquivos de origem.

---

## Fonte dos dados

Credit Card Approval Prediction — Kaggle

https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction/data

Projeto desenvolvido exclusivamente para fins acadêmicos, no âmbito da Pós-Tech FIAP — Data Analytics.
