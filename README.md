# Repositório de Algoritmos de Busca Heurística

Este repositório contém implementações de algoritmos de busca heurística desenvolvidas em Python, utilizando as bibliotecas **NumPy** e **Matplotlib** para cálculos numéricos e visualização de gráficos. Os algoritmos são aplicados a diferentes problemas de otimização com foco em maximização ou minimização de funções-objetivo.

## Estrutura do Repositório

O repositório é organizado da seguinte maneira:

algoritmos/
│
├── hill_climbing/
│   ├── solucao_1.py
│   ├── solucao_2.py
│   └── ...
│
├── busca_local_aleatoria/
│   ├── solucao_1.py
│   ├── solucao_2.py
│   └── ...
│
├── busca_aleatoria_global/
│   ├── solucao_1.py
│   ├── solucao_2.py
│   └── ...
│
├── tempera_simulada/
│   ├── problema_8_rainhas.py
│
└── algoritmo_genetico/
    ├── populacao_canonica.py
    └── populacao_ponto_fluante.py


### 1. Hill Climbing, Busca Local Aleatória e Busca Aleatória Global

As três primeiras pastas (hill_climbing, busca_local_aleatoria, busca_aleatoria_global) contêm 8 soluções para diferentes problemas de otimização. Cada solução é configurada para:

- **Funções-objetivo**: Pode ser tanto de maximização quanto de minimização, dependendo da aplicação.
- **Limites de variáveis independentes**: Definidos de acordo com cada problema.
- **Número máximo de interações**: 1000 interações por execução.
- **Critério de parada antecipada**: Se o algoritmo rodar 100 interações consecutivas sem encontrar uma solução melhor, ele é interrompido.
- **Execução**: Cada algoritmo é executado R = 100 vezes. Os valores obtidos são armazenados e utilizados para calcular a **moda** das soluções.
- **Visualização**: Cada solução gera um gráfico para melhor visualização dos resultados.

### 2. Têmpera Simulada

A pasta **tempera_simulada** contém uma implementação da Têmpera Simulada para resolver o **Problema das 8 Rainhas**. Esse problema consiste em posicionar 8 rainhas em um tabuleiro de xadrez de forma que nenhuma rainha ataque outra, resultando em 92 soluções distintas.

O algoritmo busca essas soluções otimizando o posicionamento das rainhas através de técnicas de resfriamento simuladas.

### 3. Algoritmo Genético

Na pasta **algoritmo_genetico**, há duas implementações distintas:

- **População Canônica**: Utiliza uma população canônica de indivíduos e aplica cruzamento através de um ponto de corte aleatório.
- **População com Ponto Flutuante**: Implementa uma população com valores em ponto flutuante, utilizando **torneio** como critério de seleção, **recombinação SBX** (Simulated Binary Crossover) e **mutação Gaussiana**.

Cada uma das implementações genéticas é executada **R = 100 vezes** para calcular os seguintes estatísticos:
- **Mínimo** e **máximo** valores de aptidão.
- **Média** das aptidões.
- **Desvio padrão** das soluções encontradas.

## Requisitos

Para rodar os algoritmos, você precisará das seguintes bibliotecas:

- **NumPy**: Para operações matemáticas e vetoriais.
- **Matplotlib**: Para geração de gráficos.

Você pode instalar as dependências usando o pip:

```bash
pip install numpy matplotlib
