import numpy as np
import matplotlib.pyplot as plt

# Função objetivo
def f(x1, x2):
    return (- (x2 + 47)) * np.sin(np.sqrt (np.absolute ((x1 / 2) + (x2 + 47)))) - x1 * np.sin( np.sqrt ( np.absolute (x1 - (x2 + 47))))

# Função de perturbação
def perturb(x, e):
    return np.random.uniform(low=x-e, high=x+e, size=2)

# Domínios de x1 e x2
x1_range = [-200, 20]
x2_range = [-200, 20]

# Plotar função 3D
x1_vals = np.linspace(x1_range[0], x1_range[1])
x2_vals = np.linspace(x2_range[0], x2_range[1])
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
z_vals = f(x1_grid, x2_grid)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis')

# Ponto inicial aleatório dentro do domínio
x_opt = np.array([np.random.uniform(low=x1_range[0], high=x1_range[1]), 
                  np.random.uniform(low=x2_range[0], high=x2_range[1])])
f_opt = f(x_opt[0], x_opt[1])

# Parâmetros do algoritmo
e = 0.5  # Vizinhança inicial
max_it = 1000  # Número máximo de iterações
max_viz = 50  # Número máximo de vizinhos a considerar
melhoria = True
i = 0
valores = [f_opt]

# Algoritmo Hill Climbing
while i < max_it and melhoria:
    melhoria = False
    for j in range(max_viz):
        x_cand = perturb(x_opt, e)  # Gera um novo candidato
        x_cand[0] = np.clip(x_cand[0], x1_range[0], x1_range[1])  # Manter dentro dos limites de x1
        x_cand[1] = np.clip(x_cand[1], x2_range[0], x2_range[1])  # Manter dentro dos limites de x2
        f_cand = f(x_cand[0], x_cand[1])  # Calcula o valor da função no candidato
        if f_cand < f_opt:  # Se o candidato é melhor
            x_opt = x_cand
            f_opt = f_cand
            valores.append(f_opt)
            melhoria = True
            ax.scatter(x_opt[0], x_opt[1], f_opt, color='r', marker='x')
            break  # Encerra busca na vizinhança se houver melhoria
    i += 1

# Marcar o ponto final
plt.pause(.1)
ax.scatter(x_opt[0], x_opt[1], f_opt, color='g', marker='x', s=100, linewidth=3)
plt.show()

# Plotar a evolução dos valores
plt.plot(valores)
plt.xlabel('Iterações')
plt.ylabel('Valor de f(x1, x2)')
plt.show()