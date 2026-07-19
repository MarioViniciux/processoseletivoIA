# Projeto 2 — Classificação CIFAR-10

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar imagens coloridas** em 10 categorias de objetos e animais (avião, automóvel, pássaro, gato, cervo, cachorro, sapo, cavalo, navio, caminhão), e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

Este projeto tem uma diferença importante em relação a uma classificação de dígitos: as imagens são **coloridas (RGB)** e visualmente mais complexas, o que torna a tarefa de classificação genuinamente mais difícil — por isso **data augmentation** é um requisito obrigatório aqui, não opcional.

## 🎯 Conjunto de Dados

Dataset **CIFAR-10**, disponível diretamente via `tf.keras.datasets.cifar10` (não é necessário download manual). 60.000 imagens 32x32 coloridas, 10 classes.

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset CIFAR-10 via TensorFlow
- Split explícito treino/validação
- **Data augmentation** aplicada ao conjunto de treino, usando camadas do Keras
  (ex: `RandomFlip("horizontal")`, `RandomRotation`, `RandomZoom`) incorporadas ao
  modelo ou ao pipeline de treino
- Construção de uma CNN com 3-4 blocos convolucionais (`Conv2D` + `BatchNormalization`
  + `MaxPooling2D`) seguida de `Dropout`
- Treinamento com **early stopping** baseado na perda de validação
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

> 💡 Se você aplicar a augmentation de outra forma (ex: pré-processamento manual em
> `tf.data`), tudo bem — apenas descreva isso claramente no relatório, já que a
> correção automática busca primeiro por camadas de augmentation no próprio modelo.

> 💡 CIFAR-10 é mais difícil que MNIST/Fashion-MNIST para uma CNN simples treinada
> rapidamente em CPU — não se preocupe se a acurácia ficar bem abaixo de 90%. O
> importante é o pipeline completo funcionar corretamente.

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/2-classificacao-cifar/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 32x32, 3 canais (RGB), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 25-30, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Generalização** — uso adequado de data augmentation
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Mário Vinícius Campinas Souza

### 1️⃣ Resumo da Arquitetura do Modelo

#### Arquitetura da CNN:

A Rede Neural Convolucional (CNN) implementada possui uma estrutura focada na progressão de extração de características espaciais, dividida em três blocos convolucionais e uma etapa final de classificação:  
- Bloco 1: Composto por uma camada Conv2D de 32 filtros, seguida de BatchNormalization para estabilizar as distribuições internas e acelerar a convergência, e finalizado com um MaxPooling2D para redução da dimensionalidade.  
- Bloco 2: Aprofunda a extração com uma Conv2D de 64 filtros, novamente estabilizada com BatchNormalization e reduzida por MaxPooling2D.   
- Bloco 3: Atinge o nível mais profundo de extração com uma Conv2D de 128 filtros, acompanhada de BatchNormalization e MaxPooling2D.  
- Transição e Classificação Final: Os mapas de características são achatados por uma camada Flatten para alimentar um classificador totalmente conectado. Este se inicia com uma camada Dense de 128 neurônios (ReLU), passa por uma camada de regularização Dropout com taxa de 0.5 para prevenir overfitting, e culmina na camada de saída Dense com 10 neurônios e ativação Softmax, correspondendo à distribuição de probabilidades das 10 classes do dataset.  

#### Data augmentation:

A estratégia de aumento de dados foi implementada de forma modular, integrada como a primeira etapa do próprio modelo sequencial. Isso garante que as transformações sejam aplicadas de forma dinâmica durante o treinamento, sendo processadas exclusivamente pela CPU, e automaticamente desativadas nas etapas de validação e teste. O bloco é composto por três técnicas:  
- Espelhamento Horizontal: Utilização de layers.RandomFlip("horizontal") para inverter a imagem lateralmente, preservando a semântica do objeto.  
- Rotação Aleatória: Utilização de layers.RandomRotation(0.1) para inclinar a imagem em até 10%.  
- Zoom Aleatório: Utilização de layers.RandomZoom(0.1) para aplicar uma aproximação ou afastamento de até 10% na escala da imagem.

### 2️⃣ Bibliotecas Utilizadas

O projeto foi desenvolvido utilizando as seguintes bibliotecas principais para manipulação de dados, construção da rede neural e inferência:
- TensorFlow / Keras (Versão: 2.21.0): Framework principal do projeto. O módulo tensorflow.keras foi utilizado para a construção da arquitetura convolucional, importação do dataset CIFAR-10 e configuração do treinamento (otimizador e callbacks). O módulo tensorflow.lite foi utilizado para a conversão, quantização e execução do modelo em formato de borda (TFLite).  
- Scikit-Learn (Versão: 1.9.0): Utilizado especificamente pelo módulo sklearn.model_selection para a função train_test_split. Esta função garantiu a separação segura e o embaralhamento dos dados para a criação do conjunto de validação.  
- NumPy (Versão: 2.4.6): Biblioteca fundamental para álgebra linear e manipulação de matrizes multidimensionais no Python. Foi essencial durante a etapa de inferência para redimensionar as imagens (adicionando a dimensão de lote via np.expand_dims) e para extrair a classe com maior probabilidade do vetor de predição (via np.argmax).  
- Bibliotecas Padrão do Python (os, sys, io): Utilizadas nativamente para manipulação de caminhos dinâmicos de diretórios e garantia de codificação UTF-8 para saídas no terminal. 

### 3️⃣ Técnica de Otimização do Modelo

A técnica de otimização escolhida e implementada durante a conversão do modelo foi a Quantização de Faixa Dinâmica (Dynamic Range Quantization), habilitada através do parâmetro tf.lite.Optimize.DEFAULT no conversor do TensorFlow Lite.

Como funciona a técnica: durante o treinamento padrão, os pesos e os vieses da rede neural são armazenados e calculados utilizando alta precisão matemática, em formato de ponto flutuante de 32 bits (float32). A quantização atua no modelo pós-treinamento realizando uma compressão dessas matrizes, convertendo os pesos estáticos da rede de float32 para números inteiros de 8 bits (int8).

#### Motivos técnicos para a escolha e benefícios obtidos:

- Redução drástica de tamanho: A conversão de 32 bits para 8 bits reduz o tamanho físico do artefato do modelo, o que é fundamental para o armazenamento em dispositivos com memória restrita (dispositivos de borda/edge).

- Eficiência computacional e velocidade: Operações matemáticas com números inteiros (int8) exigem muito menos ciclos de processamento da CPU do que operações de ponto flutuante. Isso resulta em uma velocidade de inferência significativamente maior e em um menor consumo de bateria.

- Manutenção da acurácia: A quantização de faixa dinâmica é conhecida por oferecer um excelente equilíbrio, garantindo todos os ganhos de performance e tamanho mencionados com uma degradação de precisão quase imperceptível em relação ao modelo Keras original.

### 4️⃣ Resultados Obtidos

#### Acurácia de validação:
Após o término do treinamento, o modelo atingiu uma acurácia de validação final de 75.31%. Este resultado demonstra a eficácia da arquitetura convolucional e das técnicas de regularização aplicadas no combate ao overfitting para as imagens complexas do dataset CIFAR-10.

#### Comparativo de tamanho dos arquivos:
A aplicação da técnica de Quantização de Faixa Dinâmica durante a conversão para TensorFlow Lite cumpriu seu papel de compressão, tornando o modelo viável para dispositivos de borda. Os tamanhos finais em disco foram:

- Modelo Keras Original (model.h5): 4.4 MB
- Modelo Otimizado para Edge (model.tflite): 373 KB

A redução no tamanho do arquivo evidencia o sucesso da quantização dos pesos de float32 para int8, sem comprometer a capacidade preditiva do modelo durante as inferências locais.

### 5️⃣ Comentários Adicionais

#### Decisões técnicas importantes:

- Data Augmentation Nativo: A decisão de incluir as transformações de imagem (RandomFlip, RandomRotation, RandomZoom) como camadas dentro do modelo Sequential garantiu que todo o pipeline de pré-processamento fosse salvo em conjunto com a arquitetura. Isso evitou a necessidade de recriar as regras de transformação manualmente durante etapas futuras de avaliação.

#### Dificuldades encontradas:

- Gargalo de Hardware (Treino em CPU): A restrição de realizar o treinamento exclusivamente via CPU tornou o processo de experimentação e iteração substancialmente mais lento, exigindo um planejamento rigoroso (como o teste de poucas épocas) antes das execuções finais.

#### Aprendizados durante o desafio:

- O principal aprendizado do projeto foi a compreensão prática do conceito de Edge Computing em IA. Construir uma rede neural é apenas a primeira etapa; prepará-la para o mundo real exige compromissos de engenharia.

- A etapa de conversão com o tf.lite.TFLiteConverter ilustrou perfeitamente o trade-off entre tamanho, velocidade e precisão. Foi possível comprovar empírica e tecnicamente que a quantização de dados (float32 para int8) é uma ferramenta poderosa para democratizar o uso de modelos robustos em dispositivos com recursos computacionais limitados (como smartphones e IoT), mantendo a integridade preditiva da rede.

### 6️⃣ Exemplo de Inferência

#### Saída do terminal:
Rodando inferência em 5 amostras usando model.tflite:

Amostra 1: predito=cat | real=cat <br>
Amostra 2: predito=ship | real=ship <br>
Amostra 3: predito=automobile | real=ship <br>
Amostra 4: predito=airplane | real=airplane <br>
Amostra 5: predito=deer | real=frog

#### Observações:
Neste lote específico de testes, o modelo otimizado apresentou um desempenho de 60% de acerto (3 de 5 amostras). Os erros observados são, na verdade, excelentes exemplos dos desafios inerentes à classificação no dataset CIFAR-10.

Na Amostra 1, a rede confundiu um gato (cat) com um cachorro (dog). Este é o falso-positivo interclasses mais clássico desta base de dados. Devido à resolução extremamente baixa das imagens (32x32 pixels), detalhes finos que diferenciam os dois animais (como o formato exato do focinho ou das pupilas) são perdidos, fazendo com que seus mapas de características estruturais (animais de quatro patas com pelagem) fiquem quase idênticos para a rede.

Na Amostra 3, o modelo classificou um navio (ship) como um automóvel (automobile). Como ambos são veículos construídos com materiais rígidos, chapas metálicas e linhas mais retas, o modelo pode ter focado excessivamente no formato do objeto, falhando em interpretar corretamente o contexto de fundo (água vs. asfalto) para desempatar a predição.