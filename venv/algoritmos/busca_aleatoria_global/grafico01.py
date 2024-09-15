import numpy as np
import matplotlib.pyplot as plt

# Define a função objetivo
def f(x):
    return x[0]**2 + x[1]**2

# Gera um candidato aleatório usando a distribuição uniforme
def gera_candidato(limites):
    return np.random.uniform(low=limites[:, 0], high=limites[:, 1])

# Busca Aleatória Global com parada antecipada
def busca_aleatoria_global(limites, max_iter=10000, t=100):
    # Inicializa a melhor solução (x_opt) e seu valor de f(x_opt)
    x_opt = gera_candidato(limites)
    f_opt = f(x_opt)
    
    sem_melhoria = 0  # Contador de iterações sem melhoria
    valores_f = [f_opt]  # Lista para armazenar os valores de f(x) ao longo das iterações
    for i in range(max_iter):
        # Gera um novo candidato aleatório
        x_cand = gera_candidato(limites)
        f_cand = f(x_cand)

        # Verifica se o novo candidato é melhor
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

    return np.round(x_opt, 3), np.round(f_opt, 3), valores_f

# Definição dos limites para as variáveis x1 e x2
limites = np.array([[-100, 100], [-100, 100]])

# Definição do grid para plotar a função 3D
x1_vals = np.linspace(limites[0, 0], limites[0, 1], 100)
x2_vals = np.linspace(limites[1, 0], limites[1, 1], 100)
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
z_vals = f([x1_grid, x2_grid])

# Plotar a superfície da função
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis', alpha=0.7)

# Executa o algoritmo GRS
x_opt, f_opt, valores_f = busca_aleatoria_global(limites, max_iter=10000, t=100)

# Marcar o ponto inicial e o ponto final
ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', s=100, label="Ótimo encontrado")
plt.legend()

# Exibir o gráfico 3D
plt.show()

# Plotar a evolução dos valores de f(x)
plt.plot(valores_f)
plt.xlabel('Iterações')
plt.ylabel('Valor de f(x1, x2)')
plt.title('Evolução da função objetivo')
plt.show()

print(f"Melhor solução encontrada: {x_opt}")
print(f"Valor da função objetivo no ótimo: {f_opt}")
