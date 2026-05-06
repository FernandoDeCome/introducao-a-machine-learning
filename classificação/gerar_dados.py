"""Gera os datasets de classificação binária e multiclasse."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1) Classificação binária — Aprovação de empréstimo
#    Features: renda_mensal, score_credito, divida_atual, idade
#    Alvo:     emprestimo_aprovado (0 = recusado, 1 = aprovado)
# ---------------------------------------------------------------------------
N1 = 400
renda_mensal = rng.uniform(1500, 25000, N1).round(2)
score_credito = rng.integers(300, 851, N1)
divida_atual = rng.uniform(0, 50000, N1).round(2)
idade = rng.integers(20, 70, N1)

# Score do "credor": combinação linear das variáveis com pesos plausíveis,
# convertida em probabilidade de aprovação via função logística.
z = (
    -6.6
    + 0.00018 * renda_mensal
    + 0.0085 * score_credito
    - 0.000050 * divida_atual
    + 0.012 * (idade - 40)
    + rng.normal(0, 0.6, N1)  # ruído
)
prob_aprovacao = 1.0 / (1.0 + np.exp(-z))
ruido = rng.uniform(0, 1, N1)
emprestimo_aprovado = (ruido < prob_aprovacao).astype(int)

df_binario = pd.DataFrame(
    {
        "renda_mensal": renda_mensal,
        "score_credito": score_credito,
        "divida_atual": divida_atual,
        "idade": idade,
        "emprestimo_aprovado": emprestimo_aprovado,
    }
)
df_binario.to_csv(
    "/Users/fernando.come/Documents/Alura/Classificação/dados_aprovacao_emprestimo.csv",
    index=False,
)
print(
    f"Binário: {len(df_binario)} linhas | aprovados: "
    f"{df_binario['emprestimo_aprovado'].sum()} "
    f"({df_binario['emprestimo_aprovado'].mean():.1%})"
)


# ---------------------------------------------------------------------------
# 2) Classificação multiclasse — Risco de Crédito
#    Features: renda_mensal, score_credito, divida_atual, idade,
#              tempo_emprego_anos, num_dependentes
#    Alvo:     risco_credito (baixo / medio / alto)
# ---------------------------------------------------------------------------
N2 = 600
renda_mensal_m = rng.uniform(1500, 30000, N2).round(2)
score_credito_m = rng.integers(300, 851, N2)
divida_atual_m = rng.uniform(0, 60000, N2).round(2)
idade_m = rng.integers(20, 70, N2)
tempo_emprego_anos = rng.uniform(0, 30, N2).round(1)
num_dependentes = rng.integers(0, 6, N2)

# "Pontuação" de saúde financeira — quanto maior, menor o risco.
pontuacao = (
    0.00010 * renda_mensal_m
    + 0.0050 * score_credito_m
    - 0.000060 * divida_atual_m
    + 0.040 * tempo_emprego_anos
    - 0.180 * num_dependentes
    + 0.012 * (idade_m - 40)
    + rng.normal(0, 0.30, N2)  # ruído
)

# Faz cortes em quantis para garantir as três classes balanceadas.
q_baixo, q_alto = np.quantile(pontuacao, [0.33, 0.66])

risco = np.where(
    pontuacao <= q_baixo,
    "alto",
    np.where(pontuacao <= q_alto, "medio", "baixo"),
)

df_multi = pd.DataFrame(
    {
        "renda_mensal": renda_mensal_m,
        "score_credito": score_credito_m,
        "divida_atual": divida_atual_m,
        "idade": idade_m,
        "tempo_emprego_anos": tempo_emprego_anos,
        "num_dependentes": num_dependentes,
        "risco_credito": risco,
    }
)
df_multi.to_csv(
    "/Users/fernando.come/Documents/Alura/Classificação/dados_risco_credito.csv",
    index=False,
)
print(
    f"Multiclasse: {len(df_multi)} linhas | distribuição: "
    f"{df_multi['risco_credito'].value_counts().to_dict()}"
)


# ---------------------------------------------------------------------------
# 3) Classificação binária complexa — Churn de Telecom
#    Features (mistura de numéricas e categóricas):
#       idade, tempo_cliente_meses, genero, tipo_contrato, tipo_internet,
#       valor_mensal, valor_total, num_servicos_extra, suporte_tecnico,
#       pagamento_automatico, forma_pagamento, num_reclamacoes_12m,
#       atrasos_pagamento_12m, uso_dados_gb
#    Alvo: churn (0 = permaneceu, 1 = cancelou)
# ---------------------------------------------------------------------------
N3 = 3000

idade_t = rng.integers(18, 80, N3)
tempo_cliente_meses = rng.integers(1, 73, N3)
genero = rng.choice(["M", "F"], N3, p=[0.49, 0.51])

# Contrato é fortemente ligado ao tempo de cliente — quem está há muito tempo
# tende a estar em planos mais longos.
contrato_probs = np.where(
    tempo_cliente_meses < 12,
    0,  # mensal mais provável
    np.where(tempo_cliente_meses < 36, 1, 2),
)
tipo_contrato = np.empty(N3, dtype=object)
for i, base in enumerate(contrato_probs):
    if base == 0:
        tipo_contrato[i] = rng.choice(
            ["mensal", "anual", "bianual"], p=[0.7, 0.2, 0.1]
        )
    elif base == 1:
        tipo_contrato[i] = rng.choice(
            ["mensal", "anual", "bianual"], p=[0.35, 0.45, 0.2]
        )
    else:
        tipo_contrato[i] = rng.choice(
            ["mensal", "anual", "bianual"], p=[0.15, 0.35, 0.5]
        )

tipo_internet = rng.choice(
    ["fibra", "dsl", "nenhuma"], N3, p=[0.5, 0.35, 0.15]
)

# Valor mensal depende do tipo de internet e dos serviços extras.
num_servicos_extra = rng.integers(0, 8, N3)
base_internet = np.where(
    tipo_internet == "fibra", 90,
    np.where(tipo_internet == "dsl", 55, 25),
)
valor_mensal = (
    base_internet
    + 8 * num_servicos_extra
    + rng.normal(0, 8, N3)
).round(2)
valor_mensal = np.clip(valor_mensal, 20, 200)

# Valor total acumulado ~ valor_mensal * tempo_cliente, com algum ruído.
valor_total = (
    valor_mensal * tempo_cliente_meses * rng.uniform(0.85, 1.05, N3)
).round(2)

suporte_tecnico = rng.choice(["sim", "nao"], N3, p=[0.4, 0.6])
pagamento_automatico = rng.choice(["sim", "nao"], N3, p=[0.55, 0.45])
forma_pagamento = rng.choice(
    ["cartao_credito", "boleto", "pix", "debito_automatico"],
    N3,
    p=[0.35, 0.25, 0.15, 0.25],
)

num_reclamacoes_12m = rng.poisson(1.2, N3).clip(0, 15)
atrasos_pagamento_12m = rng.poisson(0.9, N3).clip(0, 12)
uso_dados_gb = (rng.gamma(2.0, 50, N3)).round(1).clip(0, 500)

# Score do churn: combinação de fatores.
# Variáveis indicadoras (one-hot mental).
contrato_score = np.where(
    tipo_contrato == "mensal", 1.4,
    np.where(tipo_contrato == "anual", -0.4, -1.2),
)
internet_score = np.where(
    tipo_internet == "fibra", 0.6,
    np.where(tipo_internet == "dsl", 0.0, -0.5),
)
suporte_score = np.where(suporte_tecnico == "nao", 0.45, -0.35)
pgto_auto_score = np.where(pagamento_automatico == "nao", 0.40, -0.30)
forma_score = np.where(
    forma_pagamento == "boleto", 0.30,
    np.where(forma_pagamento == "debito_automatico", -0.30, 0.0),
)

z3 = (
    -1.6
    + contrato_score
    + internet_score
    + suporte_score
    + pgto_auto_score
    + forma_score
    - 0.022 * tempo_cliente_meses          # mais tempo → menos churn
    + 0.012 * (valor_mensal - 70)          # mais caro → mais churn
    - 0.0035 * (idade_t - 40)              # mais velho → ligeiramente menos churn
    + 0.18 * num_reclamacoes_12m           # reclamações empurram churn
    + 0.22 * atrasos_pagamento_12m         # atrasos empurram churn
    - 0.05 * num_servicos_extra            # quanto mais serviços, mais "preso"
    + 0.0008 * (uso_dados_gb - 100)        # uso muito alto → leve churn
    + rng.normal(0, 0.55, N3)              # ruído individual
)
prob_churn = 1.0 / (1.0 + np.exp(-z3))
ruido3 = rng.uniform(0, 1, N3)
churn = (ruido3 < prob_churn).astype(int)

df_churn = pd.DataFrame(
    {
        "idade": idade_t,
        "tempo_cliente_meses": tempo_cliente_meses,
        "genero": genero,
        "tipo_contrato": tipo_contrato,
        "tipo_internet": tipo_internet,
        "valor_mensal": valor_mensal,
        "valor_total": valor_total,
        "num_servicos_extra": num_servicos_extra,
        "suporte_tecnico": suporte_tecnico,
        "pagamento_automatico": pagamento_automatico,
        "forma_pagamento": forma_pagamento,
        "num_reclamacoes_12m": num_reclamacoes_12m,
        "atrasos_pagamento_12m": atrasos_pagamento_12m,
        "uso_dados_gb": uso_dados_gb,
        "churn": churn,
    }
)
df_churn.to_csv(
    "/Users/fernando.come/Documents/Alura/Classificação/dados_churn_telecom.csv",
    index=False,
)
print(
    f"Churn Telecom: {len(df_churn)} linhas | "
    f"churners: {df_churn['churn'].sum()} ({df_churn['churn'].mean():.1%})"
)
