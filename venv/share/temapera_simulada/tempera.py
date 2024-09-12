import numpy as np
import time

def f(x):
    num_attacks = 0
    n = len(x)
    for i in range(n):
        for j in range(i+1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                num_attacks += 1
    return num_attacks

def perturb(x):
    x_new = x.copy()
    i = np.random.randint(0, len(x))
    delta = np.random.choice([-1, 1])
    x_new[i] = (x_new[i] + delta) % len(x)
    return x_new

def update_temperature_exponential(T, beta=0.9):
    return T * beta

# Algoritmo Têmpera Simulada
def simulated_annealing(T_initial, max_iter):
    T = T_initial
    x_opt = np.random.permutation(8)  # Solução inicial aleatória
    f_opt = f(x_opt)
    i = 0
    f_otimos = [f_opt]

    start_time = time.time()

    while i < max_iter and f_opt != 0:
        x_cand = perturb(x_opt)
        f_cand = f(x_cand)
        p_ij = np.exp(-(f_cand - f_opt) / T)
        
        if f_cand < f_opt or p_ij >= np.random.uniform(0, 1):
            x_opt = x_cand
            f_opt = f_cand

        f_otimos.append(f_opt)
        T = update_temperature_exponential(T)  # Escalonamento exponencial
        i += 1

    end_time = time.time()
    execution_time = end_time - start_time

    return x_opt, f_opt, execution_time

T_initial = 1000
max_iter = 10000

x_final, f_final, exec_time = simulated_annealing(T_initial, max_iter)
print(f"Melhor solução: {x_final}, Ataques: {f_final}, Tempo de execução: {exec_time} segundos")
