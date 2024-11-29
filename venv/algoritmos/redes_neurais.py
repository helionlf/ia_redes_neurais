import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Carregar os dados
data = np.loadtxt('venv/data/spiral.csv', delimiter=',')

# Visualização dos dados
plt.scatter(data[data[:, 2] == 1, 0], data[data[:, 2] == 1, 1], color='green', edgecolors='black', label='Classe +1')
plt.scatter(data[data[:, 2] == -1, 0], data[data[:, 2] == -1, 1], color='red', edgecolors='black',  label='Classe -1')
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.legend()
plt.show()

x = np.array(data[:, :2])

N,p = x.shape

X = x.T
X = np.concatenate((
    -np.ones((1,N)),
    X)
)

Y = np.array(data[:, 2]).reshape(N, 1)
Y = Y.T

def sign(u):
    return 1 if u >= 0 else -1

def perceptron_simples(X, Y, w, N, p, lr):
    erro = True
    max_epoch = 1
    epoca = 0
    while erro and epoca <= max_epoch:
        erro = False
        for t in range(N):
            x_t = X[:, t].reshape(p + 1, 1)
            u_t = (w.T @ x_t)[0, 0]
            y_t = sign(u_t)
            d_t = float(Y[0, t])
            e_t = d_t - y_t
            w = w + (lr * e_t * x_t) / 2
            if y_t != d_t:
                erro = True
        epoca += 1
    return w

def EQM(X,Y,w):
    p_1,N = X.shape
    eq = 0
    for t in range(N):
        x_t = X[:,t].reshape(p_1,1)
        u_t = w.T@x_t
        d_t = Y[0,t]
        eq += (d_t-u_t[0,0])**2
    return eq/(2*N)

def adaline(X, Y, w, N, p, lr):
    pr = 1e-5
    EQM1 = 1
    EQM2 = 0
    max_epoch = 1
    epochs = 0
    hist = []
    while epochs < max_epoch and abs(EQM1-EQM2) > pr:
        EQM1 = EQM(X,Y,w)
        hist.append(EQM1)
        for t in range(N):
            x_t = X[:,t].reshape(p+1,1)
            u_t = w.T@x_t
            d_t = Y[0,t]
            e_t = d_t - u_t
            w = w + lr*e_t*x_t
        epochs+=1
        EQM2 = EQM(X,Y,w)      
    hist.append(EQM2)
    return w

def sigmoid(u):
    return 1 / (1 + np.exp(-u))

def derivada_sigmoid(u):
    return u * (1 - u)

def mlp(X, Y, p, neuronios_camada_oculta, C, lr, epocas_maximas):
    w_entrada_oculta = np.random.randn(neuronios_camada_oculta, p + 1) * 0.01
    b_oculta = np.zeros((neuronios_camada_oculta, 1))
    w_oculta_saida = np.random.randn(C, neuronios_camada_oculta) * 0.01
    b_saida = np.zeros((C, 1))

    for epoca in range(epocas_maximas):
        for t in range(X.shape[1]):  # Para cada amostra
            # Entrada da amostra com bias
            x_t = X[:, t].reshape(p + 1, 1)

            # Saída desejada
            d_t = Y[:, t].reshape(C, 1)

            # Forward pass: Camada oculta
            u_oculta = w_entrada_oculta @ x_t + b_oculta
            z_oculta = sigmoid(u_oculta)

            # Forward pass: Camada de saída
            u_saida = w_oculta_saida @ z_oculta + b_saida
            z_saida = sigmoid(u_saida)

            # Erro na saída
            e_t = d_t - z_saida

            # Backpropagation
            grad_saida = e_t * derivada_sigmoid(z_saida)
            grad_oculta = (w_oculta_saida.T @ grad_saida) * derivada_sigmoid(z_oculta)

            # Atualização dos pesos e biases
            w_oculta_saida += lr * grad_saida @ z_oculta.T
            b_saida += lr * grad_saida
            w_entrada_oculta += lr * grad_oculta @ x_t.T
            b_oculta += lr * grad_oculta

    return w_entrada_oculta, b_oculta, w_oculta_saida, b_saida

def testar_mlp(X_teste, w_entrada_oculta, b_oculta, w_oculta_saida, b_saida):
    u_oculta = w_entrada_oculta @ X_teste + b_oculta
    z_oculta = sigmoid(u_oculta)
    u_saida = w_oculta_saida @ z_oculta + b_saida
    z_saida = sigmoid(u_saida)
    return np.argmax(z_saida, axis=0)

def converter_saida_continua(y_continuo):
    return np.where(y_continuo > 0.5, 1, -1)

def calcular_metricas(y_pred, y_true):
    vp = np.sum((y_pred == 1) & (y_true == 1))  
    vn = np.sum((y_pred == -1) & (y_true == -1)) 
    fp = np.sum((y_pred == 1) & (y_true == -1))  
    fn = np.sum((y_pred == -1) & (y_true == 1)) 

    acuracia = (vp + vn) / len(y_true)
    sensibilidade = vp / (vp + fn) if (vp + fn) > 0 else 0
    especificidade = vn / (vn + fp) if (vn + fp) > 0 else 0

    return acuracia, sensibilidade, especificidade

# def matriz_confusao(y_pred, y_true):
#     vp = np.sum((y_pred == 1) & (y_true == 1))  
#     vn = np.sum((y_pred == -1) & (y_true == -1)) 
#     fp = np.sum((y_pred == 1) & (y_true == -1))  
#     fn = np.sum((y_pred == -1) & (y_true == 1)) 
    
#     conf_matrix = np.array([[vp, fn], 
#                             [fp, vn]])
#     return conf_matrix

# def plot_matriz_confusao(conf_matrix, titulo):
#     plt.figure(figsize=(6, 4))
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['+1', '-1'], yticklabels=['+1', '-1'])
#     plt.title(titulo)
#     plt.xlabel("Classe prevista")
#     plt.ylabel("Classe real")
#     plt.show()

# def plot_curva_aprendizado(hist, titulo):
#     plt.plot(hist)
#     plt.title(titulo)
#     plt.xlabel("Épocas")
#     plt.ylabel("Erro Quadrático Médio (EQM)")
#     plt.show()

acuracias_perceptron, sensibilidades_perceptron, especificidades_perceptron = [], [], []
acuracias_adaline, sensibilidades_adaline, especificidades_adaline = [], [], []
acuracias_mlp, sensibilidades_mlp, especificidades_mlp = [], [], []

res_perceptron = np.empty((0, 3))
res_adaline = np.empty((0, 3))
res_mlp = np.empty((0, 3))

lr = 0.1
C = 1
R = 500

# Monte Carlo
for i in range(R):
    indices = np.random.permutation(N)
    X, Y = X[:, indices], Y[:, indices]

    split = int(0.8 * N)
    X_treino, Y_treino = X[:, :split], Y[:, :split]
    X_teste, Y_teste = X[:, split:], Y[:, split:]

    w = np.random.random_sample((3, 1)) - 0.5

    # Treinar o perceptron simples
    w_final = perceptron_simples(X_treino, Y_treino, w, X_treino.shape[1], p, lr)
    y_pred = np.sign(w_final.T @ X_teste).flatten()

    # Calcular métricas para o perceptron simples
    acuracia, sensibilidade, especificidade = calcular_metricas(y_pred, Y_teste.flatten())
    acuracias_perceptron.append(acuracia)
    sensibilidades_perceptron.append(sensibilidade)
    especificidades_perceptron.append(especificidade)

    rodada_metrics = np.array([[acuracia, sensibilidade, especificidade]])
    res_perceptron = np.concatenate((res_perceptron, rodada_metrics), axis=0)

    # Treinar o adaline
    w_final = adaline(X_treino, Y_treino, w, X_treino.shape[1], p, lr)
    y_pred = np.sign(w_final.T @ X_teste).flatten()

    # Calcular métricas para o adaline
    acuracia, sensibilidade, especificidade = calcular_metricas(y_pred, Y_teste.flatten())
    acuracias_adaline.append(acuracia)
    sensibilidades_adaline.append(sensibilidade)
    especificidades_adaline.append(especificidade)

    rodada_metrics = np.array([[acuracia, sensibilidade, especificidade]])
    res_adaline = np.concatenate((res_adaline, rodada_metrics), axis=0)

    # Treinar o MLP
    w_entrada_oculta, b_oculta, w_oculta_saida, b_saida = mlp( X_treino, Y_treino, p, 5, C, lr, 1)
    predicoes = testar_mlp(X_teste, w_entrada_oculta, b_oculta, w_oculta_saida, b_saida)
    y_pred = converter_saida_continua(predicoes)

    # Calcular métricas para o MLP
    acuracia, sensibilidade, especificidade = calcular_metricas(y_pred, Y_teste.flatten())
    acuracias_mlp.append(acuracia)
    sensibilidades_mlp.append(sensibilidade)
    especificidades_mlp.append(especificidade)

    rodada_metrics = np.array([[acuracia, sensibilidade, especificidade]])
    res_mlp = np.concatenate((res_mlp, rodada_metrics), axis=0)

# Resultados finais

# Resultados perceptron simples
rss_mean = np.mean(res_perceptron, axis=0)
rss_std = np.std(res_perceptron, axis=0)
rss_min = np.min(res_perceptron, axis=0)
rss_max = np.max(res_perceptron, axis=0)

print("Métricas para perceptron simples:")
print("Média de acurácias, sencibilidade e especificidade repectivamente: ", rss_mean)
print("Desvio-padrão de acurácias, sencibilidade e especificidade repectivamente: ", rss_std)
print("Mínimo, sencibilidade e especificidade repectivamente: ", rss_min)
print("Máximo de acurácias, sencibilidade e especificidade repectivamente: ", rss_max)

# Resultados adaline
rss_mean = np.mean(res_adaline, axis=0)
rss_std = np.std(res_adaline, axis=0)
rss_min = np.min(res_adaline, axis=0)
rss_max = np.max(res_adaline, axis=0)

print("\nMétricas para adaline:")
print("Média de acurácias, sencibilidade e especificidade repectivamente: ", rss_mean)
print("Desvio-padrão de acurácias, sencibilidade e especificidade repectivamente: ", rss_std)
print("Mínimo, sencibilidade e especificidade repectivamente: ", rss_min)
print("Máximo de acurácias, sencibilidade e especificidade repectivamente: ", rss_max)

# Resultados mlp
rss_mean = np.mean(res_mlp, axis=0)
rss_std = np.std(res_mlp, axis=0)
rss_min = np.min(res_mlp, axis=0)
rss_max = np.max(res_mlp, axis=0)

print("\nMétricas para mlp:")
print("Média de acurácias, sencibilidade e especificidade repectivamente: ", rss_mean)
print("Desvio-padrão de acurácias, sencibilidade e especificidade repectivamente: ", rss_std)
print("Mínimo, sencibilidade e especificidade repectivamente: ", rss_min)
print("Máximo de acurácias, sencibilidade e especificidade repectivamente: ", rss_max)

# # Melhores e piores casos para Perceptron
# melhor_idx_perceptron = np.argmax(acuracias_perceptron)
# pior_idx_perceptron = np.argmin(acuracias_perceptron)

# # Melhores e piores casos para Adaline
# melhor_idx_adaline = np.argmax(acuracias_adaline)
# pior_idx_adaline = np.argmin(acuracias_adaline)

# melhor_perceptron = (acuracias_perceptron[melhor_idx_perceptron], sensibilidades_perceptron[melhor_idx_perceptron], especificidades_perceptron[melhor_idx_perceptron])
# pior_perceptron = (acuracias_perceptron[pior_idx_perceptron], sensibilidades_perceptron[pior_idx_perceptron], especificidades_perceptron[pior_idx_perceptron])

# melhor_adaline = (acuracias_adaline[melhor_idx_adaline], sensibilidades_adaline[melhor_idx_adaline], especificidades_adaline[melhor_idx_adaline])
# pior_adaline = (acuracias_adaline[pior_idx_adaline], sensibilidades_adaline[pior_idx_adaline], especificidades_adaline[pior_idx_adaline])

# # Matrizes de confusão
# y_pred_melhor = np.sign(res_perceptron[melhor_idx_perceptron].flatten())
# conf_matriz_melhor = matriz_confusao(y_pred_melhor, Y_teste.flatten())
# plot_matriz_confusao(conf_matriz_melhor, "Matriz de Confusão - Melhor Perceptron")

# y_pred_pior = np.sign(res_perceptron[pior_idx_perceptron].flatten())
# conf_matriz_pior = matriz_confusao(y_pred_pior, Y_teste.flatten())
# plot_matriz_confusao(conf_matriz_pior, "Matriz de Confusão - Pior Perceptron")

# y_pred_melhor = np.sign(res_adaline[melhor_idx_adaline].flatten())
# conf_matriz_melhor = matriz_confusao(y_pred_melhor, Y_teste.flatten())
# plot_matriz_confusao(conf_matriz_melhor, "Matriz de Confusão - Melhor Adaline")

# y_pred_pior = np.sign(res_adaline[pior_idx_adaline].flatten())
# conf_matriz_pior = matriz_confusao(y_pred_pior, Y_teste.flatten())
# plot_matriz_confusao(conf_matriz_pior, "Matriz de Confusão - Pior Adaline")