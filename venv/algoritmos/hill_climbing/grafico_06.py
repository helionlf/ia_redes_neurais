import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def calcula_moda(valores):
    contagem = Counter(valores)
    moda = contagem.most_common(1)[0][0]
    return moda

# Função objetivo
def f(x1, x2):
    return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1

# Função de perturbação 
def perturb(x, e):
    return np.random.uniform(low=x-e, high=x+e, size=2)

# Ddomínios de x1 e x2
x1_range = [-1, 3]
x2_range = [-1, 3]

# Plotar função 3D
x1_vals = np.linspace(x1_range[0], x1_range[1])
x2_vals = np.linspace(x2_range[0], x2_range[1])
x1_grid, x2_grid = np.meshgrid(x1_vals, x2_vals)
z_vals = f(x1_grid, x2_grid)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1_grid, x2_grid, z_vals, cmap='viridis')

# Parâmetros
e = .1  # Vizinhança inicial
max_it = 10000  # Número máximo de iterações
max_viz = 50  # Número máximo de vizinhos a considerar
t = 100  # Número de iterações sem melhoria para parada antecipada
R = 100  # Número de rodadas
valores_finais = []  # Armazena as melhores soluções de cada rodada

# Algoritmo Hill Climbing
def hill_climbing(x_opt, f_opt):
    valores = [f_opt]
    melhoria = True
    cont_naoMelhorias = 0
    i = 0

    while i < max_it and melhoria:
        melhoria = False
        for j in range(max_viz):
            x_cand = perturb(x_opt, e)  # Gera um novo candidato
            x_cand[0] = np.clip(x_cand[0], x1_range[0], x1_range[1])  # Mantém dentro dos limites de x1
            x_cand[1] = np.clip(x_cand[1], x2_range[0], x2_range[1])  # Mantém dentro dos limites de x2
            f_cand = f(x_cand[0], x_cand[1])  # Calcula o valor da função no candidato
            
            if f_cand > f_opt:  # Se o candidato é melhor, atualiza
                x_opt = x_cand
                f_opt = f_cand
                valores.append(f_opt)
                melhoria = True
                cont_naoMelhorias = 0  # Reset no contador de melhorias
                break
        
        if not melhoria:
            cont_naoMelhorias += 1
            if cont_naoMelhorias >= t:  # Parada antecipada por falta de melhoria
                break
        
        i += 1
    
    return f_opt

# Executa o algoritmo R vezes
for _ in range(R):
    # Ponto inicial - domínio inferior
    x_opt = np.array([x1_range[0], x2_range[0]])
    f_opt = f(x_opt[0], x_opt[1])
    
    melhor_solucao = hill_climbing(x_opt, f_opt)
    valores_finais.append(round(melhor_solucao, 3))

plt.show()

# resultados
print("Soluções finais:", valores_finais)
print("Moda das soluções finais:", calcula_moda(valores_finais))
