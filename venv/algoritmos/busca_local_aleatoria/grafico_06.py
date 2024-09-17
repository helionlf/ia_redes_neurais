import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Definindo a função a ser minimizada
def func(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1


def gera_candidato_local(limites, x_atual):
    perturbacao = np.random.uniform(low=-1, high=1, size=x_atual.shape)
    candidato = x_atual + perturbacao
    
    # Garante que o candidato esteja dentro dos limites
    for i in range(len(candidato)):
        if candidato[i] < limites[i, 0]:
            candidato[i] = limites[i, 0]
        elif candidato[i] > limites[i, 1]:
            candidato[i] = limites[i, 1]
    
    return candidato

# Função de busca aleatória local para minimização com critério de parada antecipada
def random_search(bounds, max_iters=10000, t_no_improve=100):
    best = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1])
    best_eval = func(best[0], best[1])
    no_improve_counter = 0  # Contador de iterações sem melhoria

    for i in range(max_iters):
        # Gera ponto aleatório no espaço de busca
        candidate = gera_candidato_local(bounds, best)
        
        # Avalia o ponto
        candidate_eval = func(candidate[0], candidate[1])
        
        # Verifica se é o melhor ponto encontrado (menor valor)
        if candidate_eval > best_eval:
            best, best_eval = candidate, candidate_eval
            no_improve_counter = 0  # Reseta o contador quando há melhoria
        else:
            no_improve_counter += 1  # Incrementa o contador se não há melhoria
        
        # Critério de parada antecipada: se não houver melhora em 't_no_improve' iterações consecutivas
        if no_improve_counter >= t_no_improve:
            break

    return best_eval

# Função para rodar o algoritmo R vezes e encontrar o valor mais frequente
def execute_multiple_runs(bounds, R=100, max_iters=10000, t_no_improve=100):
    results = []
    
    for _ in range(R):
        best_value = random_search(bounds, max_iters=max_iters, t_no_improve=t_no_improve)
        results.append(round(best_value, 3))  # Armazena o valor encontrado com 3 casas decimais
    
    # Encontrando o valor mais frequente
    most_common = Counter(results).most_common(1)[0][0]
    
    return results, most_common

# Limites de busca (x1 e x2 entre [-100, 100])
bounds = np.array([[-1, 3], [-1, 3]])

# Definição do grid para plotar a função 3D
x1_vals = np.linspace(bounds[0, 0], bounds[0, 1], 100)
x2_vals = np.linspace(bounds[1, 0], bounds[1, 1], 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
z_vals = func(x1_grid, x2_grid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis', alpha=0.7)
plt.show()

# Executando a busca aleatória local várias vezes
R = 100  # Número de execuções
results, resultado_frequentista = execute_multiple_runs(bounds, R=R)

# Exibindo o resultado frequentista e o melhor ponto encontrado
print(f"Resultados finais: {results}")
print(f"Resultado Frequentista após {R} execuções: {resultado_frequentista}")
