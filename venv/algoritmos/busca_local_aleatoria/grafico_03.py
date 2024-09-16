import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Definindo a função a ser minimizada
def func(x1, x2):
    return -20 * np.exp(- (0.2 * np.sqrt(0.5 * (x1**2 + x2**2)))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)

# Função de busca aleatória local para minimização com critério de parada
def random_search(bounds, max_iters=10000, t_no_improve=100):
    best = None
    best_eval = float('inf')  # Inicializa o melhor valor como infinito (para minimização)
    no_improve_counter = 0  # Contador de iterações sem melhoria

    for i in range(max_iters):
        # Gera ponto aleatório no espaço de busca
        candidate = np.random.uniform(bounds[:, 0], bounds[:, 1])
        
        # Avalia o ponto
        candidate_eval = func(candidate[0], candidate[1])
        
        # Verifica se é o melhor ponto encontrado (menor valor)
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval
            no_improve_counter = 0  # Reseta o contador quando há melhoria
        else:
            no_improve_counter += 1  # Incrementa o contador se não há melhoria
        
        # Critério de parada antecipada: se não houver melhora em 't_no_improve' iterações consecutivas
        if no_improve_counter >= t_no_improve:
            break

    return best, best_eval

# Função para rodar o algoritmo R vezes e encontrar o valor mais frequente
def execute_multiple_runs(bounds, R=100, max_iters=10000, t_no_improve=100):
    results = []
    
    for _ in range(R):
        best_point, best_value = random_search(bounds, max_iters=max_iters, t_no_improve=t_no_improve)
        results.append(round(best_value, 3))  # Armazena o valor encontrado com 3 casas decimais
    
    # Encontrando o valor mais frequente
    most_common = Counter(results).most_common(1)[0][0]
    
    return most_common

# Limites de busca (x1 e x2 entre [-8, 8])
bounds = np.array([[-8, 8], [-8, 8]])

# Executando a busca aleatória local várias vezes
R = 100  # Número de execuções
resultado_frequentista = execute_multiple_runs(bounds, R=R)

# Gerando valores para x1 e x2 no intervalo especificado
x1 = np.linspace(-8, 8, 400)
x2 = np.linspace(-8, 8, 400)

# Criando a grade de valores de x1 e x2 para a função
X1, X2 = np.meshgrid(x1, x2)
Z = func(X1, X2)

# Criando o gráfico 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plotando a superfície
surf = ax.plot_surface(X1, X2, Z, cmap='rainbow', edgecolor='none')

# Executando a busca aleatória para encontrar o melhor ponto
best_point, best_value = random_search(bounds)

# Plotando o ponto mínimo encontrado
ax.scatter(best_point[0], best_point[1], best_value, color='black', s=100, label='Mínimo Encontrado')

# Adicionando o gráfico de contorno na base
ax.contour(X1, X2, Z, levels=20, cmap='rainbow', linestyles="solid", offset=np.min(Z))

# Adicionando rótulos e título
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Gráfico da Função com Busca Aleatória Local')

# Exibindo a barra de cores
fig.colorbar(surf)

# Exibindo o gráfico
plt.legend()
plt.show()

# Exibindo o resultado frequentista e o melhor ponto encontrado
print(f"Resultado Frequentista após {R} execuções: {resultado_frequentista}")
print(f"Melhor Ponto: {best_point}, Melhor Valor: {best_value}")
