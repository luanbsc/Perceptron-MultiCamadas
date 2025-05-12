import random
import os
import numpy as np
import matplotlib.pyplot as plt

# Para gerar o gráfico
gfx_iteracoes = []
gfx_EQM = []

def sigmoid(x):
    return 1 / (1 + np.exp(-(B) * x))

def sigmoid_derivada(x):
    return B * sigmoid(x) * (1 - sigmoid(x))

# Topologia
entradas = 4
neuronios = 20
saidas = 2

EQM_media = []
acuracia_media = []
epocas_media = []


print("Topologia escolhida >>> ", entradas, "-", neuronios, "-", saidas, sep="")
# Looping para percorrer todos os arquivos
for numero_arquivo in range(1, 11):
    # Construção do vetor de pesos inical
    w = []
    for camada in range(2):
        for i in range(neuronios - (neuronios-saidas)*camada):
            lista_pesos = []
            if camada == 0:
                for j in range(entradas + 1):
                    lista_pesos.append(random.uniform(0, 1))
            else:
                for j in range(neuronios + 1):
                    lista_pesos.append(random.uniform(0, 1))
            w.append(lista_pesos)
    #print('Vetor de pesos inicial: ', w)
    n = 0.1

    Erro = 0.000001
    B = 0.5
    EQM_ant = 9999999999
    EQM_atual = 0
    iteracoes = 0
    EQM = 0

    # Importação das entradas a partir dos arquivos
    nome_arquivo = f'datasets/iris/iris-10dobscv-{numero_arquivo}tra.dat'
    print("Calculando resultados do arquivo ", nome_arquivo, "...", sep="")
    with open(nome_arquivo, 'r') as file:
        linhas = file.readlines()
    T = linhas[9:25]
    linhas = linhas[24:]


    # Treinamento
    while abs(EQM_atual - EQM_ant) > Erro:
        EQM_ant = EQM_atual
        for p in linhas:
            dados = p.strip().split(', ')
            x = list(map(float, dados[:4]))
            x.insert(0, -1)
            if dados[4] == "Iris-setosa":
                d = [0, 0]
            elif dados[4] == "Iris-versicolor":
                d = [0, 1]
            elif dados[4] == "Iris-virginica":
                d = [1, 0]

            I = []
            Y = []
            gradiente = []
            gradiente.append([])
            gradiente.append([])

            for camada in range(2):              #For para cada camada
                i_somatorio = []
                y_valores = []
                for i in range(neuronios - (neuronios-saidas)*camada):
                    if camada == 0:
                        if i == 0:
                            y_valores.append(-1)
                        i_somatorio.append(w[i][0] * x[0] + w[i][1] * x[1] + w[i][2] * x[2] + w[i][3] * x[3] + w[i][4] * x[4])
                        y_valores.append(sigmoid(i_somatorio[i]))
                    else:
                        i_somatorio.append(w[i+neuronios][0] * Y[0][0] + w[i+neuronios][1] * Y[0][1] + w[i+neuronios][2] * Y[0][2] + w[i+neuronios][3] * Y[0][3] + w[i+neuronios][4] * Y[0][4] + w[i+neuronios][5] * Y[0][5])
                        y_valores.append(sigmoid(i_somatorio[i]))
                I.append(i_somatorio)
                Y.append(y_valores)

            # Calculo do gradiente da camada 2
            for i in range(2):
                gradiente[1].append((d[i] - Y[1][i]) * sigmoid_derivada(I[1][i]))
            
            # Ajuste dos pesos da camada 2
            for i in range(2):
                for j in range(neuronios + 1):
                    w[i+neuronios][j] += n * gradiente[1][i] * Y[0][j]
            
            # Calculo do gradiente da camada 1
            for i in range(neuronios):
                gd = 0
                for j in range(2):
                    gd += gradiente[1][j] * w[j+neuronios][i+1]
                gradiente[0].append(gd * sigmoid_derivada(I[0][i]))
            
            # Ajuste dos pesos da camada 1
            for i in range(neuronios):
                for j in range(entradas + 1):
                    w[i][j] += n * gradiente[0][i] * x[j]
        
        # Cálculo do Erro Quadrático Médio
        eq = 0
        for i in range(saidas):
            eq += (d[i] - Y[1][i]) * (d[i] - Y[1][i])
        eq /= 2
        EQM += eq
        EQM = EQM/(iteracoes + 1)
        EQM_atual = EQM

        iteracoes += 1

        gfx_EQM.append(EQM_atual)
        gfx_iteracoes.append(iteracoes)
        #print("Iteracoes:", iteracoes, "\tEQM:", EQM_atual)

    # Validação
    linhas_teste = 0
    acertos = 0
    for p in T:
        linhas_teste += 1
        dados_teste = p.strip().split(', ')
        if dados_teste[4] == "Iris-setosa":
            d_teste = [0, 0]
        elif dados_teste[4] == "Iris-versicolor":
            d_teste = [0, 1]
        elif dados_teste[4] == "Iris-virginica":
            d_teste = [1, 0]
        x_teste = list(map(float, dados_teste[:4]))
        x_teste.insert(0, -1)
        I_teste = []
        Y_teste = []
        for camada in range(2):              #For para cada camada
            i_somatorio = []
            y_valores = []
            for i in range(neuronios - (neuronios-saidas)*camada):
                if camada == 0:
                    if i == 0:
                        y_valores.append(-1)
                    i_somatorio.append(w[i][0] * x_teste[0] + w[i][1] * x_teste[1] + w[i][2] * x_teste[2] + w[i][3] * x_teste[3] + w[i][4] * x_teste[4])
                    y_valores.append(sigmoid(i_somatorio[i]))
                else:
                    i_somatorio.append(w[i+neuronios][0] * Y_teste[0][0] + w[i+neuronios][1] * Y_teste[0][1] + w[i+neuronios][2] * Y_teste[0][2] + w[i+neuronios][3] * Y_teste[0][3] + w[i+neuronios][4] * Y_teste[0][4] + w[i+neuronios][5] * Y_teste[0][5])
                    y_valores.append(sigmoid(i_somatorio[i]))
            I_teste.append(i_somatorio)
            Y_teste.append(y_valores)
        
        resultado = []
        if Y_teste[1][0] >= 0.5:
            resultado.append(1)
        else:
            resultado.append(0)
        
        if Y_teste[1][1] >= 0.5:
            resultado.append(1)
        else:
            resultado.append(0)
        if d_teste == resultado:
            acertos += 1

    EQM_media.append(EQM)
    acuracia_media.append((acertos/linhas_teste)*100)
    epocas_media.append(iteracoes)

    #print("Porcentagem de acerto do algoritmo: ", (acertos/linhas_teste)*100, "%", sep="")


    # Plotação de gráfico para verificar precisão do algoritmo
    '''plt.figure()
    plt.plot(gfx_iteracoes, gfx_EQM)
    plt.xlabel('Iterações')
    plt.ylabel('EQM')
    plt.title(f'EQM em função das iterações - Arquivo: {numero_arquivo}tra.dat')
    plt.show()'''

print("Média do EQM:", sum(EQM_media)/len(EQM_media), "\tDesvio padrão do EQM:", np.std(EQM_media))
print("Média da acurácia:", sum(acuracia_media)/len(acuracia_media), "\tDesvio padrão da acurácia:", np.std(acuracia_media))
print("Média do número de épocas:", sum(epocas_media)/len(epocas_media), "\tDesvio padrão do número de épocas:", np.std(epocas_media))