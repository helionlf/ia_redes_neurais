import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# from collections import Counter

# def calcula_moda(valores):
#     contagem = Counter(valores)
#     moda = contagem.most_common(1)[0][0]
#     return moda

# Função objetivo
def f(x):
    return (x[0]**2 - 10 * np.cos(2 * np.pi * x[0]) + 10) + (x[1]**2 - 10 * np.cos(2 * np.pi * x[1]) + 10)

# Candidato aleatório
def gera_candidato(limites):
    return np.random.uniform(low=limites[:, 0], high=limites[:, 1])

# Busca Aleatória Global
def busca_aleatoria_global(limites, max_iter=10000, t=100):
    x_opt = gera_candidato(limites)
    f_opt = f(x_opt)
    
    sem_melhoria = 0  # Contador de iterações sem melhoria
    valores_f = [f_opt]
    
    for i in range(max_iter):
        # Novo candidato aleatório
        x_cand = gera_candidato(limites)
        f_cand = f(x_cand)

        # Verificar se o novo candidato é melhor
        if f_cand < f_opt:
            f_opt = f_cand
            x_opt = x_cand
            sem_melhoria = 0  # Reseta o contador de iterações sem melhoria
        else:
            sem_melhoria += 1

        valores_f.append(f_opt)

        # Parada antecipada se não houver melhorias em 't' iterações consecutivas
        if sem_melhoria >= t:
            break

    return np.round(f_opt, 3)

# Função para rodar R (100) execuções do GRS e armazenar os resultados
def executa_grs_varias_vezes(limites, R=100, max_iter=10000, t=100):
    solucoes = []
    
    for _ in range(R):
        f_opt = busca_aleatoria_global(limites, max_iter, t)
        solucoes.append(f_opt)
    
    # Calcula a moda das soluções
    moda_solucoes = stats.mode(solucoes)[0][0]
    
    return solucoes, np.round(moda_solucoes, 3)

# Limites para as variáveis x1 e x2
limites = np.array([[-5.12, 5.12], [-5.12, 5.12]])

# Definição do grid para plotar a função 3D
x1_vals = np.linspace(limites[0, 0], limites[0, 1], 100)
x2_vals = np.linspace(limites[1, 0], limites[1, 1], 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
z_vals = f([x1_grid, x2_grid])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis', alpha=0.7)
plt.show()

# Executa o algoritmo GRS 100 vezes
solucoes, moda_solucoes = executa_grs_varias_vezes(limites, R=100)

print(f"Moda das soluções: {moda_solucoes}")
print(f"Soluções encontradas: {solucoes}")
