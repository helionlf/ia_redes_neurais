import numpy as np

# Função de Rastrigin
def rastrigin(x, A=10):
    p = len(x)
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão
def aptidao(x):
    return rastrigin(x) + 1

# Seleção por roleta
def roleta(populacao, aptidoes):
    aptidoes_soma = np.sum(aptidoes)
    selecao_probs = aptidoes / aptidoes_soma
    return populacao[np.random.choice(range(len(populacao)), p=selecao_probs)]

# Cruzamento (a partir de um ponto aleatório)
def cruzamento(pai1, pai2):
    ponto_corte = np.random.randint(1, len(pai1) - 1)
    filho1 = np.concatenate([pai1[:ponto_corte], pai2[ponto_corte:]])
    filho2 = np.concatenate([pai2[:ponto_corte], pai1[ponto_corte:]])
    return filho1, filho2

# Mutação binária
def mutacao(individuo, taxa_mutacao=0.01):
    for i in range(len(individuo)):
        if np.random.rand() < taxa_mutacao:
            individuo[i] = 1 - individuo[i]  # Inverte o bit
    return individuo

# Inicializar a população com representação canônica
def inicializar_populacao(tamanho_populacao, dimensoes):
    return np.random.randint(0, 2, (tamanho_populacao, dimensoes))

# Critério de convergência: número máximo de gerações
def convergencia(geracao_atual, max_geracoes):
    return geracao_atual >= max_geracoes

# Algoritmo genético canônico
def algoritmo_genetico_can(p, A=10, max_geracoes=100, taxa_recombinacao=0.9, taxa_mutacao=0.01):
    tamanho_populacao = 100
    populacao = inicializar_populacao(tamanho_populacao, p)
    geracao = 0

    while not convergencia(geracao, max_geracoes):
        aptidoes = np.array([aptidao(ind) for ind in populacao])

        # Nova população
        nova_populacao = []

        for _ in range(tamanho_populacao // 2):
            pai1 = roleta(populacao, aptidoes)
            pai2 = roleta(populacao, aptidoes)

            if np.random.rand() < taxa_recombinacao:
                filho1, filho2 = cruzamento(pai1, pai2)
            else:
                filho1, filho2 = pai1, pai2

            # Mutação
            filho1 = mutacao(filho1, taxa_mutacao)
            filho2 = mutacao(filho2, taxa_mutacao)

            nova_populacao.extend([filho1, filho2])

        populacao = np.array(nova_populacao)
        geracao += 1

    melhor_individuo = populacao[np.argmin([aptidao(ind) for ind in populacao])]
    return melhor_individuo, aptidao(melhor_individuo)


solucoes = []
for i in range(100):
    p = 20
    resultado = algoritmo_genetico_can(p)
    solucoes.append(resultado)
print("-------------------------------------------------------------------------------------------")

# Aptidões das soluções
aptidoes = np.array([resultado[1] for resultado in solucoes])

# Calculos das métricas
menor_aptidao = np.min(aptidoes)
maior_aptidao = np.max(aptidoes)
media_aptidao = np.mean(aptidoes)
desvio_padrao_aptidao = np.std(aptidoes)

print(f"Menor aptidão: {menor_aptidao}")
print(f"Maior aptidão: {maior_aptidao}")
print(f"Média de aptidão: {media_aptidao}")
print(f"Desvio padrão de aptidão: {desvio_padrao_aptidao}")
