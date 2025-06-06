<h1>🌸 Classificador MLP com Backpropagation - Dataset Iris 🌸</h1>

  <p>Este projeto implementa uma <strong>rede neural multicamada (MLP)</strong> utilizando <strong>backpropagation</strong> para classificação de flores do famoso <strong>conjunto de dados Iris</strong>. O treinamento é feito com validação cruzada estratificada (10-fold cross-validation) e os resultados são avaliados com base na <strong>acurácia</strong>, <strong>erro quadrático médio (EQM)</strong> e número de <strong>épocas de treinamento</strong>.</p>

  <h2>📚 Sobre o Dataset</h2>
  <p>O conjunto de dados Iris possui 150 amostras, cada uma representando uma flor com quatro características:</p>
  <ul>
    <li>Comprimento da sépala</li>
    <li>Largura da sépala</li>
    <li>Comprimento da pétala</li>
    <li>Largura da pétala</li>
  </ul>
  <p>As flores pertencem a uma das seguintes espécies:</p>
  <ul>
    <li><em>Iris setosa</em></li>
    <li><em>Iris versicolor</em></li>
    <li><em>Iris virginica</em></li>
  </ul>
  <p>A classificação é feita com <strong>2 neurônios de saída binária</strong>, representando as três espécies como:</p>
  <ul>
    <li>[0, 0] → <em>Iris setosa</em></li>
    <li>[0, 1] → <em>Iris versicolor</em></li>
    <li>[1, 0] → <em>Iris virginica</em></li>
  </ul>

  <h2>🧠 Topologia da Rede Neural</h2>
  <ul>
    <li><strong>Entradas:</strong> 4 (características da flor)</li>
    <li><strong>Camada oculta:</strong> 20 neurônios com ativação sigmoide</li>
    <li><strong>Saídas:</strong> 2 neurônios binários</li>
    <li><strong>Função de ativação:</strong> Sigmoide com parâmetro B = 0.5</li>
    <li><strong>Taxa de aprendizado:</strong> 0.1</li>
    <li><strong>Critério de parada:</strong> Convergência do EQM</li>
  </ul>

  <h2>🔁 Treinamento e Validação</h2>
  <p>Os dados são divididos em 10 conjuntos com validação cruzada (10-fold).</p>
  <p>A cada iteração:</p>
  <ul>
    <li>O modelo é treinado com 135 amostras.</li>
    <li>É testado com 15 amostras.</li>
  </ul>
  <p>Métricas computadas:</p>
  <ul>
    <li>EQM final médio e desvio padrão</li>
    <li>Acurácia média e desvio padrão</li>
    <li>Épocas médias e desvio padrão</li>
  </ul>

  <h2>📈 Exemplo de Resultados</h2>
  <pre>
Topologia escolhida >>> 4-20-2
Calculando resultados do arquivo datasets/iris/iris-10dobscv-1tra.dat...
...
Média do EQM: 0.0152     Desvio padrão do EQM: 0.0047
Média da acurácia: 96.0% Desvio padrão da acurácia: 2.53%
Média do número de épocas: 41.6 Desvio padrão do número de épocas: 5.32
  </pre>

  <h2>🧪 Requisitos</h2>
  <ul>
    <li>Python 3.7+</li>
    <li>Bibliotecas:
      <ul>
        <li>numpy</li>
        <li>matplotlib</li>
      </ul>
    </li>
  </ul>
  <p>Instale com:</p>
  <pre><code>pip install numpy matplotlib</code></pre>

  <h2>📁 Estrutura esperada</h2>
  <pre>
.
├── datasets/
│   └── iris/
│       ├── iris-10dobscv-1tra.dat
│       ├── iris-10dobscv-2tra.dat
│       └── ... (até iris-10dobscv-10tra.dat)
├── PerceptronMC.py
└── README.md
  </pre>

  <h2>▶️ Como Executar</h2>
  <pre><code>python PerceptronMC.py</code></pre>

  <h2>📊 Gráficos (Opcional)</h2>
  <p>Para visualizar os gráficos de EQM × Iterações, descomente o seguinte trecho do código:</p>
  <pre><code>
# plt.figure()
# plt.plot(gfx_iteracoes, gfx_EQM)
# plt.xlabel('Iterações')
# plt.ylabel('EQM')
# plt.title(f'EQM em função das iterações - Arquivo: {numero_arquivo}tra.dat')
# plt.show()
  </code></pre>

  <h2>📌 Observações</h2>
  <ul>
    <li>O código foi escrito com foco didático, ideal para estudos sobre redes neurais simples e aprendizado supervisionado.</li>
    <li>A saída binária é uma codificação alternativa à função softmax, mantendo simplicidade na implementação.</li>
  </ul>

  <h2>👨‍💻 Autor</h2>
  <p>Desenvolvido por <strong>Luan Barbosa dos Santos Costa</strong>.</p>
  <p>Sinta-se à vontade para contribuir, sugerir melhorias ou utilizar o projeto em trabalhos e estudos! 🚀</p>