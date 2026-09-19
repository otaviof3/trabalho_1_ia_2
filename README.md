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

## Execução

Instale as dependências:

```powershell
pip install -r requirements.txt

E execute o arquivo Python:
```powershell
python player_per_game_ia2.py