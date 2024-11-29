import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

#ATENÇÃO:
#Salve este algoritmo no mesmo diretório no qual a pasta chamada RecFac está.


#A tarefa nessa etapa é realizar o reconhecimento facial de 20 pessoas

#Dimensões da imagem. Você deve explorar esse tamanho de acordo com o solicitado no pdf.
dimensao = 50 #50 signica que a imagem terá 50 x 50 pixels. ?No trabalho é solicitado para que se investigue dimensões diferentes:
# 50x50, 40x40, 30x30, 20x20, 10x10 .... (tua equipe pode tentar outros redimensionamentos.)

#Criando strings auxiliares para organizar o conjunto de dados:
pasta_raiz = "venv/algoritmos/RecFac"
caminho_pessoas = [x[0] for x in os.walk(pasta_raiz)]
caminho_pessoas.pop(0)

C = 20 #Esse é o total de classes 
X = np.empty((dimensao*dimensao,0)) # Essa variável X será a matriz de dados de dimensões p x N. 
Y = np.empty((C,0)) #Essa variável Y será a matriz de rótulos (Digo matriz, pois, é solicitado o one-hot-encoding).
for i,pessoa in enumerate(caminho_pessoas):
    imagens_pessoa = os.listdir(pessoa)
    for imagens in imagens_pessoa:

        caminho_imagem = os.path.join(pessoa,imagens)
        imagem_original = cv2.imread(caminho_imagem,cv2.IMREAD_GRAYSCALE)
        imagem_redimensionada = cv2.resize(imagem_original,(dimensao,dimensao))

        #A imagem pode ser visualizada com esse comando.
        # No entanto, o comando deve ser comentado quando o algoritmo for executado
        # cv2.imshow("eita",imagem_redimensionada)
        # cv2.waitKey(0)

        #vetorizando a imagem:
        x = imagem_redimensionada.flatten()

        #Empilhando amostra para criar a matriz X que terá dimensão p x N
        X = np.concatenate((
            X,
            x.reshape(dimensao*dimensao,1)
        ),axis=1)
        
        #one-hot-encoding (A EQUIPE DEVE DESENVOLVER)
        y = -np.ones((C,1))
        y[i,0] = 1

        Y = np.concatenate((
            Y,
            y
        ),axis=1)
       
p,N=X.shape
    


# Normalização dos dados (A EQUIPE DEVE ESCOLHER O TIPO E DESENVOLVER):
def normalizar_dados(X):
    return (X - np.min(X)) / (np.max(X) - np.min(X))

# Início das rodadas de monte carlo
#Aqui podem existir as definições dos hiperparâmetros de cada modelo.







def sign(u):
    return 1 if u >= 0 else -1

def perceptron_simples(X, Y, w, N, p, lr):
    erro = True
    max_epoch = 1
    epoca = 0
    while erro and epoca <= max_epoch:
        erro = False
        for t in range(N):
            x_t = X[:, t].reshape(p, 1)
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

    # Inicialização dos pesos e biases
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
    return np.argmax(z_saida, axis=0)  # Classe com maior ativação


def calcular_metricas(y_pred, y_true):
    vp = np.sum((y_pred == 1) & (y_true == 1))  
    vn = np.sum((y_pred == -1) & (y_true == -1)) 
    fp = np.sum((y_pred == 1) & (y_true == -1))  
    fn = np.sum((y_pred == -1) & (y_true == 1)) 

    acuracia = (vp + vn) / len(y_true)
    sensibilidade = vp / (vp + fn) if (vp + fn) > 0 else 0
    especificidade = vn / (vn + fp) if (vn + fp) > 0 else 0

    return acuracia, sensibilidade, especificidade


acuracias_perceptron, sensibilidades_perceptron, especificidades_perceptron = [], [], []
acuracias_adaline, sensibilidades_adaline, especificidades_adaline = [], [], []

res_perceptron = np.empty((0, 3))
res_adaline = np.empty((0, 3))

lr = 0.1
R = 50

for i in range(R):
    indices = np.random.permutation(N)
    X = normalizar_dados(X)
    X = X[:, indices]
    Y = Y[:, indices]

    split = int(0.8 * N)
    X_treino, Y_treino = X[:, :split], Y[:, :split]
    X_teste, Y_teste = X[:, split:], Y[:, split:]

    w = np.random.random_sample((p, 1)) - 0.5

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
    res_adaline = np.concatenate((res_perceptron, rodada_metrics), axis=0)

    # Treinar o MLP
    w_entrada_oculta, b_oculta, w_oculta_saida, b_saida = mlp( X_treino, Y_treino, p, 16, C, lr, 1)
    predicoes = testar_mlp(X_teste, w_entrada_oculta, b_oculta, w_oculta_saida, b_saida)



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

