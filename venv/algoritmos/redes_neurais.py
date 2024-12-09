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
    max_epoch = 50
    epoca = 0
    while erro and epoca < max_epoch:
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
    max_epoch = 50
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
            w = np.nan_to_num(w, nan=0.0, posinf=1e10, neginf=-1e10)
        epochs+=1
        EQM2 = EQM(X,Y,w)      
    hist.append(EQM2)
    return w

# Função de ativação e sua derivada
def sigmoid(u):
    u = np.clip(u, -500, 500)
    return 1 / (1 + np.exp(-u))

def derivada_sigmoid(u):
    return np.clip(u * (1 - u), 0, 1)

# Configurar os pesos da MLP
def configurar_mlp(Xtreino, Ytreino, L, qtd_neuronios, C):
    p = Xtreino.shape[0] 
    W = []

    # Inicialização dos pesos
    for l in range(L + 1):
        if l == 0: 
            W.append(np.random.uniform(-0.5, 0.5, (qtd_neuronios[l], p + 1)))
        elif l == L:
            W.append(np.random.uniform(-0.5, 0.5, (C, qtd_neuronios[l-1] + 1)))
        else: 
            W.append(np.random.uniform(-0.5, 0.5, (qtd_neuronios[l], qtd_neuronios[l-1] + 1)))

    return W

# Forward pass para o MLP
def forward(Xamostra, W, L):
    i, y = [], []
    y_atual = np.vstack([-np.ones((1, Xamostra.shape[1])), Xamostra])  

    for l in range(L + 1):
        i_atual = W[l] @ y_atual  # Calcula o input para a camada
        y_atual = sigmoid(i_atual) if l < L else i_atual  # Aplica sigmoide para camadas ocultas
        i.append(i_atual)
        if l < L:
            y_atual = np.vstack([-np.ones((1, y_atual.shape[1])), y_atual])
        y.append(y_atual)

    return i, y

# Backward para MLP
def backward(Xamostra, d, W, i, y, L, eta):
    delta = [None] * (L + 1)

    # Calcula o delta da saída
    erro_saida = d - y[L] 
    delta[L] = erro_saida * derivada_sigmoid(y[L])

    # Backpropagation para os deltas das camadas ocultas
    for l in range(L - 1, -1, -1):
        W_sem_bias = W[l + 1][:, 1:]
        delta[l] = (W_sem_bias.T @ delta[l + 1]) * derivada_sigmoid(y[l][1:, :])

    # Atualiza os pesos
    for l in range(L + 1):
        if l == 0:
            entrada = Xamostra
        else:
            entrada = y[l - 1][1:, :]

        # Adiciona o bias à entrada antes da atualização
        entrada_com_bias = np.vstack([-np.ones((1, entrada.shape[1])), entrada])
        W[l] += eta * (delta[l] @ entrada_com_bias.T)  # Atualiza os pesos
        W[l] = np.nan_to_num(W[l], nan=0.0, posinf=1e10, neginf=-1e10)
    return W

def calcular_EQM(Xtreino, Ytreino, W, L):
    EQM = 0
    N = Xtreino.shape[1]

    for t in range(N):
        xamostra = Xtreino[:, [t]]
        d = Ytreino[:, [t]]
        _, y = forward(xamostra, W, L)
        EQM += np.sum((d - y[L]) ** 2)

    return EQM / (2 * N)

# Função para treinamento do MLP
def treinar_mlp(Xtreino, Ytreino, L, qtd_neuronios, C, eta, maxEpoch, criterio_parada):
    W = configurar_mlp(Xtreino, Ytreino, L, qtd_neuronios, C)
    N = Xtreino.shape[1]
    EQM = 1
    epoch = 0

    while EQM > criterio_parada and epoch < maxEpoch:
        for t in range(N):
            xamostra = Xtreino[:, [t]]
            d = Ytreino[:, [t]]
            i, y = forward(xamostra, W, L)
            W = backward(xamostra, d, W, i, y, L, eta)

        EQM = calcular_EQM(Xtreino, Ytreino, W, L)
        epoch += 1
    return W

# Função para testar a0 MLP
def testar_mlp(Xteste, W, L):
    y_pred = []

    for t in range(Xteste.shape[1]):
        xamostra = Xteste[:, [t]]
        _, y = forward(xamostra, W, L)
        y_pred.append(np.sign(y[L]))  # Classificação por sinal

    return np.hstack(y_pred)

def calcular_metricas(y_pred, y_true):
    vp = np.sum((y_pred == 1) & (y_true == 1))  
    vn = np.sum((y_pred == -1) & (y_true == -1)) 
    fp = np.sum((y_pred == 1) & (y_true == -1))  
    fn = np.sum((y_pred == -1) & (y_true == 1)) 

    acuracia = (vp + vn) / len(y_true)
    sensibilidade = vp / (vp + fn) if (vp + fn) > 0 else 0
    especificidade = vn / (vn + fp) if (vn + fp) > 0 else 0

    return acuracia, sensibilidade, especificidade

# Calcular a matriz de confusão
def calcular_matriz_confusao(y_true, y_pred, num_classes):
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for real, pred in zip(y_true, y_pred):
        matriz[real, pred] += 1
    return matriz

# Plotar a matriz de confusão
def plotar_matriz_confusao(matriz, titulo):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(titulo)
    plt.xlabel("Predito")
    plt.ylabel("Real")
    plt.show()

# Plotar a curva de aprendizado
def plotar_curva_aprendizado(acuracias, titulo):
    plt.figure(figsize=(10, 6))
    plt.plot(acuracias, marker='o', label="Acurácia")
    plt.title(titulo)
    plt.xlabel("Rodada")
    plt.ylabel("Acurácia")
    plt.grid()
    plt.legend()
    plt.show()

def normalizar_dados(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

res_perceptron = np.empty((0, 3))
res_adaline = np.empty((0, 3))
res_mlp = np.empty((0, 3))

lr = 0.01
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

    rodada_metrics = np.array([[acuracia, sensibilidade, especificidade]])
    res_perceptron = np.concatenate((res_perceptron, rodada_metrics), axis=0)

    # Treinar o adaline
    w_final = adaline(X_treino, Y_treino, w, X_treino.shape[1], p, lr)
    y_pred = np.sign(w_final.T @ X_teste).flatten()

    # Calcular métricas para o adaline
    acuracia, sensibilidade, especificidade = calcular_metricas(y_pred, Y_teste.flatten())

    rodada_metrics = np.array([[acuracia, sensibilidade, especificidade]])
    res_adaline = np.concatenate((res_adaline, rodada_metrics), axis=0)

    # Treinar O MLP
    qtd_neuronios = [8, 4, 4, 8]  
    L = len(qtd_neuronios)
    maxEpoch = 50
    critérioParada = 1e-5
    eta = 0.01

    X_treino = normalizar_dados(X_treino)
    
    W_mlp = treinar_mlp(X_treino, Y_treino, L, qtd_neuronios, C, eta, maxEpoch, critérioParada)
    y_pred = testar_mlp(X_teste, W_mlp, L)
    

    # Calcular métricas para a MLP
    acuracia_mlp, sensibilidade_mlp, especificidade_mlp = calcular_metricas(y_pred, Y_teste.flatten())

    rodada_metrics_mlp = np.array([[acuracia_mlp, sensibilidade_mlp, especificidade_mlp]])
    res_mlp = np.concatenate((res_mlp, rodada_metrics_mlp), axis=0)

# Resultados finais

# Resultados perceptron simples
rss_mean = np.mean(res_perceptron, axis=0)
rss_std = np.std(res_perceptron, axis=0)
rss_min = np.min(res_perceptron, axis=0)
rss_max = np.max(res_perceptron, axis=0)

print("Métricas para perceptron simples:")
print("Média de acurácias, sencibilidade e especificidade repectivamente: ", rss_mean)
print("Desvio-padrão de acurácias, sencibilidade e especificidade repectivamente: ", rss_std)
print("Mínimo de acurácias, sencibilidade e especificidade repectivamente: ", rss_min)
print("Máximo de acurácias, sencibilidade e especificidade repectivamente: ", rss_max)

# Resultados adaline
rss_mean = np.mean(res_adaline, axis=0)
rss_std = np.std(res_adaline, axis=0)
rss_min = np.min(res_adaline, axis=0)
rss_max = np.max(res_adaline, axis=0)

print("\nMétricas para adaline:")
print("Média de acurácias, sencibilidade e especificidade repectivamente: ", rss_mean)
print("Desvio-padrão de acurácias, sencibilidade e especificidade repectivamente: ", rss_std)
print("Mínimo de acurácias, sencibilidade e especificidade repectivamente: ", rss_min)
print("Máximo de acurácias, sencibilidade e especificidade repectivamente: ", rss_max)

# Resultados mlp
rss_mean = np.mean(res_mlp, axis=0)
rss_std = np.std(res_mlp, axis=0)
rss_min = np.min(res_mlp, axis=0)
rss_max = np.max(res_mlp, axis=0)

print("\nMétricas para mlp:")
print("Média de acurácias, sencibilidade e especificidade repectivamente: ", rss_mean)
print("Desvio-padrão de acurácias, sencibilidade e especificidade repectivamente: ", rss_std)
print("Mínimo de acurácias, sencibilidade e especificidade repectivamente: ", rss_min)
print("Máximo de acurácias, sencibilidade e especificidade repectivamente: ", rss_max)
