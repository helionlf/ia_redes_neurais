import numpy as np

# Função de Rastrigin
def rastrigin(x, A=10):
    p = len(x)
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão
def aptidao(x):
    return rastrigin(x) + 1

# Seleção por torneio
def torneio(populacao, aptidoes, k=3):
    competidores = np.random.choice(range(len(populacao)), k, replace=False)
    aptidoes_competidores = [aptidoes[i] for i in competidores]
    vencedor = competidores[np.argmin(aptidoes_competidores)]
    return populacao[vencedor]

# Simulated Binary Crossover (SBX)
def crossover_sbx(pai1, pai2, eta=10):
    filho1 = np.empty_like(pai1)
    filho2 = np.empty_like(pai2)
    
    for i in range(len(pai1)):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2 * u) ** (1 / (eta + 1))
        else:
            beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
        
        filho1[i] = 0.5 * ((1 + beta) * pai1[i] + (1 - beta) * pai2[i])
        filho2[i] = 0.5 * ((1 - beta) * pai1[i] + (1 + beta) * pai2[i])
    
    return filho1, filho2

# Mutação Gaussiana
def mutacao_gaussiana(individuo, taxa_mutacao=0.01, sigma=0.1):
    for i in range(len(individuo)):
        if np.random.rand() < taxa_mutacao:
            individuo[i] += np.random.normal(0, sigma)
    return individuo

# Inicializar a população com ponto flutuante
def inicializar_populacao_pflutuante(tamanho_populacao, dimensoes, limite_inferior=-10, limite_superior=10):
    return np.random.uniform(limite_inferior, limite_superior, (tamanho_populacao, dimensoes))

# Critério de convergência: número máximo de gerações
def convergencia(geracao_atual, max_geracoes):
    return geracao_atual >= max_geracoes

# Algoritmo genético com ponto flutuante
def algoritmo_genetico_pflutuante(p, A=10, max_geracoes=100, taxa_recombinacao=0.9, taxa_mutacao=0.01):
    tamanho_populacao = 100
    populacao = inicializar_populacao_pflutuante(tamanho_populacao, p)
    geracao = 0

    while not convergencia(geracao, max_geracoes):
        aptidoes = np.array([aptidao(ind) for ind in populacao])

        # Nova população
        nova_populacao = []

        for _ in range(tamanho_populacao // 2):
            pai1 = torneio(populacao, aptidoes)
            pai2 = torneio(populacao, aptidoes)

            if np.random.rand() < taxa_recombinacao:
                filho1, filho2 = crossover_sbx(pai1, pai2)
            else:
                filho1, filho2 = pai1, pai2

            # Mutação Gaussiana
            filho1 = mutacao_gaussiana(filho1, taxa_mutacao)
            filho2 = mutacao_gaussiana(filho2, taxa_mutacao)

            nova_populacao.extend([filho1, filho2])

        populacao = np.array(nova_populacao)
        geracao += 1

    melhor_individuo = populacao[np.argmin([aptidao(ind) for ind in populacao])]
    return melhor_individuo, aptidao(melhor_individuo)


solucoes = []
for i in range(100):
    p = 20
    resultado = algoritmo_genetico_pflutuante(p)
    solucoes.append(resultado)
print("-------------------------------------------------------------------------------------------")

# Extraímos as aptidões das soluções
aptidoes = np.array([resultado[1] for resultado in solucoes])

# Calculamos as métricas solicitadas
menor_aptidao = np.min(aptidoes)
maior_aptidao = np.max(aptidoes)
media_aptidao = np.mean(aptidoes)
desvio_padrao_aptidao = np.std(aptidoes)

print(f"Menor aptidão: {menor_aptidao}")
print(f"Maior aptidão: {maior_aptidao}")
print(f"Média de aptidão: {media_aptidao}")
print(f"Desvio padrão de aptidão: {desvio_padrao_aptidao}")
