# Clusterização de Jogadores da NBA

## Descrição

Este projeto utiliza Machine Learning não supervisionado para agrupar
jogadores da NBA com base em suas médias estatísticas.

O algoritmo utilizado é o K-Means, com quatro clusters na configuração final.

## Objetivo

Identificar grupos de jogadores com perfis estatisticamente semelhantes
e permitir a análise de um jogador hipotético a partir de suas médias.

## Dataset

O projeto utiliza o csv `Player Per Game.csv` do dataset [https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats/data).

O csv contém estatísticas de jogadores de basquete, incluindo pontos,
rebotes, assistências, minutos, arremessos, aproveitamentos e outras métricas.

## Tipo de Machine Learning

Aprendizado não supervisionado, utilizando clustering.

## Tecnologias

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Metodologia

1. Filtragem dos registros da NBA;
2. Seleção da temporada analisada;
3. Tratamento de jogadores que atuaram por mais de uma equipe;
4. Seleção das variáveis estatísticas;
5. Tratamento dos valores ausentes;
6. Padronização dos atributos;
7. Teste de diferentes valores de K;
8. Treinamento do K-Means com quatro clusters;
9. Avaliação e visualização dos agrupamentos;
10. Teste de um jogador hipotético.

## Estrutura esperada

```
projeto/
├── data/
│   └── Player Per Game.csv
├── outputs/              (gerada automaticamente)
├── reports/
│   └── relatorio.pdf
├── player_per_game_ia2.py
└── requirements.txt
```

Baixe o arquivo `Player Per Game.csv` do dataset linkado acima e coloque-o
na pasta `data/` antes de executar.

## Execução

Instale as dependências:

```powershell
pip install -r requirements.txt
```

Execute o script:

```powershell
python player_per_game_ia2.py
```

## Resultados gerados

Ao final da execução, a pasta `outputs/` conterá os CSVs e gráficos gerados
(clusters por jogador, perfil dos clusters, avaliação por K, entre outros).
A lista completa dos arquivos e sua descrição está no relatório técnico,
disponível em `reports/`.

## Reprodutibilidade

O script utiliza `random_state = 42` em todos os modelos, garantindo que
execuções repetidas produzam exatamente os mesmos clusters. Os parâmetros
principais (arquivo de entrada, temporada analisada, número de clusters e
faixa de K testada) estão centralizados no topo do arquivo
`player_per_game_ia2.py`.