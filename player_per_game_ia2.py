# ============================================================
# PROJETO IA 2 - CLUSTERIZAÇÃO DE JOGADORES DA NBA
# Algoritmo: K-Means
# ============================================================
#
# Objetivo:
# Agrupar jogadores da NBA de uma temporada com base
# em suas estatísticas de desempenho.
#
# IMPORTANTE:
# O algoritmo NÃO recebe a posição do jogador como variável
# para criar os grupos.
#
# O K-Means encontra jogadores estatisticamente semelhantes.
#
# Neste projeto serão utilizados 4 clusters.
#
# ============================================================


# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import os
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA


warnings.filterwarnings("ignore")


# ============================================================
# 2. CONFIGURAÇÕES DO PROJETO
# ============================================================

# Nome do arquivo CSV
ARQUIVO = "Player Per Game.csv"

# Temporada que será analisada
TEMPORADA = 2026

# Número de clusters que queremos
NUM_CLUSTERS = 4

# Para avaliar outros valores de K
K_MIN = 2
K_MAX = 12

# Seed para garantir que os resultados sejam reproduzíveis
RANDOM_STATE = 42


# ============================================================
# 3. VERIFICAR SE O ARQUIVO EXISTE
# ============================================================

if not os.path.exists(ARQUIVO):

    raise FileNotFoundError(
        f"O arquivo '{ARQUIVO}' não foi encontrado.\n"
        "Coloque o arquivo CSV na mesma pasta do arquivo Python."
    )


print("=" * 70)
print("PROJETO IA 2 - CLUSTERIZAÇÃO DE JOGADORES DA NBA")
print("=" * 70)

print(f"\nArquivo utilizado: {ARQUIVO}")

print(f"Temporada analisada: {TEMPORADA}")

print(f"Número de clusters escolhido: {NUM_CLUSTERS}")


# ============================================================
# 4. CARREGAR O DATASET
# ============================================================

df = pd.read_csv(ARQUIVO)


print("\n" + "=" * 70)
print("4. CARREGAMENTO DOS DADOS")
print("=" * 70)

print(
    f"\nQuantidade de registros: {df.shape[0]}"
)

print(
    f"Quantidade de colunas: {df.shape[1]}"
)


# ============================================================
# 5. VISÃO GERAL
# ============================================================

print("\nPrimeiras linhas do dataset:")

print(
    df.head()
)


print("\nColunas:")

for coluna in df.columns:

    print("-", coluna)


# ============================================================
# 6. FILTRAR SOMENTE A NBA
# ============================================================

df_nba = df[
    df["lg"] == "NBA"
].copy()


print("\n" + "=" * 70)
print("6. FILTRO DA NBA")
print("=" * 70)

print(
    f"\nRegistros da NBA: {len(df_nba)}"
)


# ============================================================
# 7. FILTRAR A TEMPORADA
# ============================================================

df_temporada = df_nba[
    df_nba["season"] == TEMPORADA
].copy()


print("\n" + "=" * 70)
print(f"7. TEMPORADA {TEMPORADA}")
print("=" * 70)

print(
    f"\nRegistros antes do tratamento: "
    f"{len(df_temporada)}"
)


# ============================================================
# 8. VERIFICAR JOGADORES COM MÚLTIPLOS REGISTROS
# ============================================================
#
# Alguns jogadores jogaram por mais de um time na temporada.
#
# O dataset possui uma linha "2TM", que representa o total
# daquele jogador na temporada.
#
# Para evitar que um mesmo jogador apareça várias vezes,
# vamos:
#
# 1. verificar quais jogadores possuem 2TM;
# 2. manter somente a linha 2TM desses jogadores;
# 3. manter normalmente os jogadores que possuem apenas um
#    time.
#
# ============================================================

contagem_jogadores = (
    df_temporada
    .groupby("player")
    .size()
)


jogadores_multiplos = contagem_jogadores[
    contagem_jogadores > 1
]


print(
    f"\nJogadores com mais de um registro: "
    f"{len(jogadores_multiplos)}"
)


# Jogadores que possuem uma linha 2TM
jogadores_2tm = set(

    df_temporada.loc[
        df_temporada["team"] == "2TM",
        "player"
    ]

)


# Remover os registros dos times individuais
# quando o jogador possui uma linha 2TM
df_temporada = df_temporada[
    ~(
        df_temporada["player"].isin(jogadores_2tm)
        &
        (df_temporada["team"] != "2TM")
    )
].copy()


print(
    f"Registros depois do tratamento: "
    f"{len(df_temporada)}"
)


print(
    f"Jogadores únicos: "
    f"{df_temporada['player'].nunique()}"
)


# ============================================================
# 9. REMOVER EVENTUAIS DUPLICATAS
# ============================================================

duplicatas = df_temporada.duplicated(
    subset=["player", "season"]
).sum()


print("\n" + "=" * 70)
print("9. DUPLICATAS")
print("=" * 70)

print(
    f"\nDuplicatas jogador + temporada: "
    f"{duplicatas}"
)


if duplicatas > 0:

    df_temporada = (
        df_temporada
        .drop_duplicates(
            subset=["player", "season"],
            keep="first"
        )
    )


# ============================================================
# 10. FEATURES UTILIZADAS NO K-MEANS
# ============================================================
#
# Não utilizamos:
#
# player
# player_id
# season
# team
# lg
#
# porque são identificadores.
#
# Também NÃO utilizamos "pos".
#
# A posição será mantida apenas para análise posterior.
#
# ============================================================

features = [

    "mp_per_game",

    "fg_per_game",
    "fga_per_game",
    "fg_percent",

    "x3p_per_game",
    "x3pa_per_game",
    "x3p_percent",

    "ft_per_game",
    "fta_per_game",
    "ft_percent",

    "trb_per_game",

    "ast_per_game",

    "stl_per_game",

    "blk_per_game",

    "tov_per_game",

    "pts_per_game"

]


print("\n" + "=" * 70)
print("10. FEATURES")
print("=" * 70)

print(
    "\nEstatísticas utilizadas:"
)

for feature in features:

    print(
        "-",
        feature
    )


# ============================================================
# 11. VERIFICAR VALORES AUSENTES
# ============================================================

print("\n" + "=" * 70)
print("11. VALORES AUSENTES")
print("=" * 70)


missing = (
    df_temporada[features]
    .isnull()
    .sum()
)


print(
    "\nValores ausentes por variável:"
)

print(
    missing
)


print(
    "\nTotal de valores ausentes:",
    missing.sum()
)


# ============================================================
# 12. SEPARAR AS FEATURES
# ============================================================

X = df_temporada[
    features
].copy()


# ============================================================
# 13. TRATAMENTO DOS VALORES AUSENTES
# ============================================================
#
# Usaremos a mediana.
#
# ============================================================

imputer = SimpleImputer(
    strategy="median"
)


X_imputado = imputer.fit_transform(
    X
)


X_imputado = pd.DataFrame(

    X_imputado,

    columns=features,

    index=df_temporada.index

)


print(
    "\nValores ausentes após tratamento:",
    X_imputado.isnull().sum().sum()
)


# ============================================================
# 14. PADRONIZAÇÃO
# ============================================================
#
# O K-Means trabalha com distância.
#
# Portanto, precisamos colocar todas as variáveis em uma
# escala comparável.
#
# ============================================================

scaler = StandardScaler()


X_scaled = scaler.fit_transform(
    X_imputado
)


print(
    "\nDados padronizados com StandardScaler."
)


# ============================================================
# 15. TESTAR DIFERENTES VALORES DE K
# ============================================================
#
# Mesmo que o projeto utilize 10 clusters, vamos testar
# diferentes valores para verificar como o comportamento
# do modelo muda.
#
# K = 2 até K = 12
#
# ============================================================

print("\n" + "=" * 70)
print("15. TESTE DE DIFERENTES VALORES DE K")
print("=" * 70)


resultados_k = []


for k in range(
    K_MIN,
    K_MAX + 1
):

    modelo_teste = KMeans(

        n_clusters=k,

        random_state=RANDOM_STATE,

        n_init=10

    )


    labels_teste = (
        modelo_teste
        .fit_predict(X_scaled)
    )


    silhouette = silhouette_score(

        X_scaled,

        labels_teste

    )


    inertia = modelo_teste.inertia_


    resultados_k.append({

        "k": k,

        "silhouette": silhouette,

        "inertia": inertia

    })


    print(

        f"K = {k:2d} | "
        f"Silhouette = {silhouette:.4f} | "
        f"Inércia = {inertia:.2f}"

    )


resultados_k = pd.DataFrame(
    resultados_k
)


# ============================================================
# 16. SALVAR RESULTADOS DOS TESTES
# ============================================================

resultados_k.to_csv(

    "avaliacao_k.csv",

    index=False,

    encoding="utf-8-sig"

)


print(
    "\nArquivo criado: avaliacao_k.csv"
)


# ============================================================
# 17. MOSTRAR O MELHOR SILHOUETTE
# ============================================================

melhor_linha = (
    resultados_k
    .loc[
        resultados_k["silhouette"].idxmax()
    ]
)


print("\n" + "=" * 70)
print("17. SILHOUETTE")
print("=" * 70)


print(
    f"\nMaior Silhouette encontrado:"
)


print(
    f"K = {int(melhor_linha['k'])}"
)


print(
    f"Silhouette = "
    f"{melhor_linha['silhouette']:.4f}"
)


print(
    f"\nK utilizado no projeto: "
    f"{NUM_CLUSTERS}"
)


# ============================================================
# 18. GRÁFICO DO SILHOUETTE
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    resultados_k["k"],

    resultados_k["silhouette"],

    marker="o"

)


plt.axvline(

    NUM_CLUSTERS,

    linestyle="--",

    label=f"K escolhido = {NUM_CLUSTERS}"

)


plt.xticks(
    resultados_k["k"]
)


plt.xlabel(
    "Número de clusters (K)"
)


plt.ylabel(
    "Silhouette Score"
)


plt.title(
    "Silhouette Score para diferentes valores de K"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    "silhouette_por_k.png",

    dpi=300

)


plt.show()


# ============================================================
# 19. GRÁFICO DO COTOVELO
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(

    resultados_k["k"],

    resultados_k["inertia"],

    marker="o"

)


plt.axvline(

    NUM_CLUSTERS,

    linestyle="--",

    label=f"K escolhido = {NUM_CLUSTERS}"

)


plt.xticks(
    resultados_k["k"]
)


plt.xlabel(
    "Número de clusters (K)"
)


plt.ylabel(
    "Inércia"
)


plt.title(
    "Método do cotovelo - K-Means"
)


plt.legend()


plt.grid(
    True,
    alpha=0.3
)


plt.tight_layout()


plt.savefig(

    "cotovelo_kmeans.png",

    dpi=300

)


plt.show()


# ============================================================
# 20. TREINAR O MODELO FINAL COM 10 CLUSTERS
# ============================================================

modelo_final = KMeans(

    n_clusters=NUM_CLUSTERS,

    random_state=RANDOM_STATE,

    n_init=10

)


clusters = (
    modelo_final
    .fit_predict(X_scaled)
)


# ============================================================
# 21. ADICIONAR OS CLUSTERS AO DATAFRAME
# ============================================================

df_resultado = (
    df_temporada.copy()
)


df_resultado["cluster"] = clusters


# ============================================================
# 22. QUANTIDADE DE JOGADORES POR CLUSTER
# ============================================================

print("\n" + "=" * 70)
print("22. TAMANHO DOS CLUSTERS")
print("=" * 70)


tamanho_clusters = (

    df_resultado[
        "cluster"
    ]

    .value_counts()

    .sort_index()

)


for cluster, quantidade in (
    tamanho_clusters.items()
):

    print(

        f"Cluster {cluster}: "
        f"{quantidade} jogadores"

    )


# ============================================================
# 23. LISTAR TODOS OS JOGADORES DE CADA CLUSTER
# ============================================================
#
# ESTA É UMA DAS PARTES MAIS IMPORTANTES.
#
# Aqui você conseguirá verificar exatamente quem pertence
# a cada grupo.
#
# ============================================================

print("\n" + "=" * 70)
print("23. JOGADORES DE CADA CLUSTER")
print("=" * 70)


for cluster in range(
    NUM_CLUSTERS
):

    jogadores_cluster = (

        df_resultado[
            df_resultado["cluster"] == cluster
        ]

        [

            [
                "player",
                "team",
                "pos",
                "pts_per_game",
                "trb_per_game",
                "ast_per_game"
            ]

        ]

        .sort_values(
            "player"
        )

    )


    print("\n")
    print("-" * 70)

    print(

        f"CLUSTER {cluster} "
        f"({len(jogadores_cluster)} jogadores)"

    )

    print("-" * 70)


    for _, jogador in (
        jogadores_cluster.iterrows()
    ):

        print(

            f"{jogador['player']} "
            f"| Time: {jogador['team']} "
            f"| Pos: {jogador['pos']} "
            f"| PTS: {jogador['pts_per_game']:.1f} "
            f"| TRB: {jogador['trb_per_game']:.1f} "
            f"| AST: {jogador['ast_per_game']:.1f}"

        )


# ============================================================
# 24. SALVAR TODOS OS JOGADORES COM SEUS CLUSTERS
# ============================================================
#
# Este será o principal arquivo para a sua análise.
#
# Você pode abrir no Excel ou Google Sheets e filtrar
# a coluna "cluster".
#
# ============================================================

colunas_saida = [

    "player",

    "player_id",

    "season",

    "team",

    "pos",

    "age",

    "g",

    "gs"

] + features + [

    "cluster"

]


df_jogadores_clusters = (

    df_resultado[
        colunas_saida
    ]

    .sort_values(
        [
            "cluster",
            "player"
        ]
    )

)


df_jogadores_clusters.to_csv(

    "jogadores_por_cluster.csv",

    index=False,

    encoding="utf-8-sig"

)


print("\n" + "=" * 70)

print(
    "Arquivo criado:"
)

print(
    "jogadores_por_cluster.csv"
)


# ============================================================
# 25. PERFIL MÉDIO DE CADA CLUSTER
# ============================================================
#
# Aqui calculamos a média das estatísticas de cada grupo.
#
# Isso permite comparar os clusters.
#
# ============================================================

perfil_clusters = (

    df_resultado

    .groupby("cluster")[features]

    .mean()

)


print("\n" + "=" * 70)
print("25. PERFIL MÉDIO DOS CLUSTERS")
print("=" * 70)


print(
    perfil_clusters.round(2)
)


# ============================================================
# 26. SALVAR PERFIL DOS CLUSTERS
# ============================================================

perfil_clusters.to_csv(

    "perfil_clusters.csv",

    encoding="utf-8-sig"

)


print(
    "\nArquivo criado: perfil_clusters.csv"
)


# ============================================================
# 27. HEATMAP DOS CLUSTERS
# ============================================================

perfil_padronizado = pd.DataFrame(

    StandardScaler().fit_transform(
        perfil_clusters
    ),

    index=perfil_clusters.index,

    columns=perfil_clusters.columns

)


plt.figure(
    figsize=(18, 8)
)


sns.heatmap(

    perfil_padronizado,

    annot=True,

    fmt=".2f",

    cmap="coolwarm",

    center=0

)


plt.title(
    "Perfil estatístico dos 10 clusters"
)


plt.xlabel(
    "Estatísticas"
)


plt.ylabel(
    "Cluster"
)


plt.tight_layout()


plt.savefig(

    "perfil_clusters_heatmap.png",

    dpi=300

)


plt.show()


# ============================================================
# 28. PCA
# ============================================================
#
# O PCA será usado somente para visualizar os clusters.
#
# O K-Means foi treinado utilizando todas as features.
#
# ============================================================

pca = PCA(
    n_components=2
)


X_pca = (
    pca
    .fit_transform(X_scaled)
)


df_resultado["PCA1"] = X_pca[:, 0]

df_resultado["PCA2"] = X_pca[:, 1]


print("\n" + "=" * 70)
print("28. PCA")
print("=" * 70)


print(

    f"\nVariância explicada pela PCA1: "
    f"{pca.explained_variance_ratio_[0] * 100:.2f}%"

)


print(

    f"Variância explicada pela PCA2: "
    f"{pca.explained_variance_ratio_[1] * 100:.2f}%"

)


print(

    f"Variância explicada pelas duas: "
    f"{pca.explained_variance_ratio_.sum() * 100:.2f}%"

)


# ============================================================
# 29. GRÁFICO DOS 10 CLUSTERS
# ============================================================

plt.figure(
    figsize=(13, 9)
)


sns.scatterplot(

    data=df_resultado,

    x="PCA1",

    y="PCA2",

    hue="cluster",

    palette="tab10",

    s=70,

    alpha=0.8

)


plt.title(

    f"Clusterização dos jogadores da NBA - "
    f"Temporada {TEMPORADA}"

)


plt.xlabel(
    "Componente Principal 1"
)


plt.ylabel(
    "Componente Principal 2"
)


plt.legend(
    title="Cluster",
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)


plt.grid(
    True,
    alpha=0.2
)


plt.tight_layout()


plt.savefig(

    "clusters_pca.png",

    dpi=300

)


plt.show()


# ============================================================
# 30. DISTÂNCIA DOS JOGADORES AO CENTRO DO CLUSTER
# ============================================================
#
# Cada cluster possui um centroide.
#
# Vamos calcular a distância de cada jogador até o centro
# do seu próprio cluster.
#
# Quanto menor a distância:
#
# -> mais próximo o jogador está do perfil central do grupo.
#
# Isso será utilizado para encontrar jogadores
# representativos.
#
# ============================================================

centroides = (
    modelo_final.cluster_centers_
)


distancias = []


for i in range(
    len(X_scaled)
):

    cluster_atual = clusters[i]


    centroide = (
        centroides[cluster_atual]
    )


    distancia = np.linalg.norm(

        X_scaled[i] - centroide

    )


    distancias.append(
        distancia
    )


df_resultado[
    "distancia_centroide"
] = distancias


# ============================================================
# 31. 5 JOGADORES MAIS REPRESENTATIVOS DE CADA CLUSTER
# ============================================================
#
# Estes jogadores são aqueles mais próximos do centro
# estatístico do seu cluster.
#
# ============================================================

print("\n" + "=" * 70)
print("31. JOGADORES REPRESENTATIVOS")
print("=" * 70)


jogadores_representativos = []


for cluster in range(
    NUM_CLUSTERS
):

    representativos = (

        df_resultado[
            df_resultado["cluster"] == cluster
        ]

        .sort_values(
            "distancia_centroide"
        )

        .head(5)

    )


    print("\n")
    print("-" * 70)

    print(
        f"CLUSTER {cluster}"
    )

    print("-" * 70)


    for _, jogador in (
        representativos.iterrows()
    ):

        print(

            f"{jogador['player']} "
            f"| Time: {jogador['team']} "
            f"| Distância: "
            f"{jogador['distancia_centroide']:.3f}"

        )


        jogadores_representativos.append({

            "cluster": cluster,

            "player": jogador["player"],

            "team": jogador["team"],

            "pos": jogador["pos"],

            "pts_per_game":
                jogador["pts_per_game"],

            "trb_per_game":
                jogador["trb_per_game"],

            "ast_per_game":
                jogador["ast_per_game"],

            "distancia_centroide":
                jogador["distancia_centroide"]

        })


df_representativos = pd.DataFrame(

    jogadores_representativos

)


# ============================================================
# 32. SALVAR JOGADORES REPRESENTATIVOS
# ============================================================

df_representativos.to_csv(

    "jogadores_representativos.csv",

    index=False,

    encoding="utf-8-sig"

)


print("\nArquivo criado:")
print(
    "jogadores_representativos.csv"
)


# ============================================================
# 33. SILHOUETTE FINAL DO MODELO COM 10 CLUSTERS
# ============================================================

silhouette_final = silhouette_score(

    X_scaled,

    clusters

)


print("\n" + "=" * 70)
print("33. AVALIAÇÃO FINAL DO MODELO")
print("=" * 70)


print(

    f"\nNúmero de clusters: "
    f"{NUM_CLUSTERS}"

)


print(

    f"Silhouette Score: "
    f"{silhouette_final:.4f}"

)


print(

    f"Inércia: "
    f"{modelo_final.inertia_:.2f}"

)


# ============================================================
# 34. SALVAR RESULTADO COMPLETO
# ============================================================

df_resultado_final = (

    df_resultado

    .sort_values(
        [
            "cluster",
            "player"
        ]
    )

)


df_resultado_final.to_csv(

    "resultado_clusterizacao_nba_2026.csv",

    index=False,

    encoding="utf-8-sig"

)


print("\nArquivo criado:")
print(
    "resultado_clusterizacao_nba_2026.csv"
)


# ============================================================
# 35. RESUMO FINAL
# ============================================================

print("\n" + "=" * 70)
print("35. RESUMO FINAL")
print("=" * 70)


print(
    f"\nTemporada analisada: {TEMPORADA}"
)


print(
    f"Jogadores analisados: "
    f"{len(df_resultado)}"
)


print(
    f"Clusters utilizados: "
    f"{NUM_CLUSTERS}"
)


print(
    f"Silhouette Score final: "
    f"{silhouette_final:.4f}"
)


print("\nQuantidade de jogadores por cluster:")


for cluster, quantidade in (
    tamanho_clusters.items()
):

    print(

        f"Cluster {cluster}: "
        f"{quantidade} jogadores"

    )


print("\n" + "=" * 70)
print("ARQUIVOS GERADOS")
print("=" * 70)


print(
    "\n1. avaliacao_k.csv"
)

print(
    "   Resultados de K = 2 até K = 12."
)


print(
    "\n2. silhouette_por_k.png"
)

print(
    "   Gráfico do Silhouette Score."
)


print(
    "\n3. cotovelo_kmeans.png"
)

print(
    "   Gráfico da inércia."
)


print(
    "\n4. perfil_clusters.csv"
)

print(
    "   Média das estatísticas de cada cluster."
)


print(
    "\n5. perfil_clusters_heatmap.png"
)

print(
    "   Heatmap para comparar os clusters."
)


print(
    "\n6. clusters_pca.png"
)

print(
    "   Visualização dos 10 clusters em 2 dimensões."
)


print(
    "\n7. jogadores_por_cluster.csv"
)

print(
    "   TODOS os jogadores + suas estatísticas + cluster."
)


print(
    "\n8. jogadores_representativos.csv"
)

print(
    "   5 jogadores mais próximos do centro de cada cluster."
)


print(
    "\n9. resultado_clusterizacao_nba_2026.csv"
)

print(
    "   Resultado completo da clusterização."
)


print("\n" + "=" * 70)
print("FIM DO PROGRAMA")
print("=" * 70)