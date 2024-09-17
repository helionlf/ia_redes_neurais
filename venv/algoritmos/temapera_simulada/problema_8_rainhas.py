import numpy as np
import time

def f(x):
    num_attacks = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            if x[i] == x[j] or abs(x[i] - x[j]) == abs(i - j):
                num_attacks += 1
    return 28 - num_attacks  # Normalização para maximizar


def perturb(x):
    x_new = x.copy()
    i = np.random.randint(0, len(x))
    delta = np.random.choice([-1, 1])
    x_new[i] = (x_new[i] + delta) % len(x)  # Garante que as rainhas fiquem em posições válidas
    return x_new


# def update_temperature_exponential(T, beta=0.9):
#     return T * beta

def update_temperature_linear(T, alpha=0.1):
    return T - alpha

# def update_temperature_log(T, i):
#     return T / np.log(i + 1)

# Algoritmo Têmpera Simulada
def simulated_annealing(T_initial, max_iter, unique_solutions):
    T = T_initial
    x_opt = np.random.permutation(8)  # Solução inicial aleatória
    f_opt = f(x_opt)
    i = 0

    start_time = time.time()

    while i < max_iter and f_opt != 28:
        x_cand = perturb(x_opt)
        f_cand = f(x_cand)
        p_ij = np.exp(-(f_cand - f_opt) / T)
        
        if f_cand > f_opt or p_ij >= np.random.uniform(0, 1):
            x_opt = x_cand
            f_opt = f_cand

        # T = update_temperature_exponential(T)
        T = update_temperature_linear(T)
        # T = update_temperature_log(T, i)
        i += 1

    end_time = time.time()
    execution_time = end_time - start_time

    # Armazena a solução se for ótima e única
    if f_opt == 28:
        unique_solutions.add(tuple(x_opt))

    return x_opt, f_opt, execution_time

T_initial = 1000
max_iter = 10000

unique_solutions = set()
while len(unique_solutions) < 92:
    x_final, f_final, exec_time = simulated_annealing(T_initial, max_iter, unique_solutions)
    print(f"Soluções encontradas: {len(unique_solutions)}/92, Última solução: {x_final} - Time: {exec_time}")
