"""
Módulo de Treinamento - Projeto CIFAR-10 Edge
Este script é responsável por carregar o dataset CIFAR-10, pré-processar as imagens,
construir uma Rede Neural Convolucional (CNN) com Data Augmentation, treinar o modelo
exclusivamente na CPU e salvar o artefato final em formato Keras (.h5).
"""

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# 1. Configuração de Hardware
# ---------------------------------------------------------

# Força o TensorFlow a utilizar apenas a CPU do sistema, escondendo qualquer 
# GPU disponível. Isso garante o cumprimento da restrição de ambiente do projeto.
tf.config.set_visible_devices([], 'GPU')

# ---------------------------------------------------------
# 2. Carregamento e Pré-processamento de Dados
# ---------------------------------------------------------

# Baixa o dataset CIFAR-10, separando automaticamente em treino e teste
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normaliza os valores dos pixels de [0, 255] para o intervalo [0, 1]
# Isso acelera e estabiliza os cálculos matemáticos (gradientes) durante o treino
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Separa 20% das imagens de treino para criar o conjunto de validação
# O random_state=42 garante que o embaralhamento será sempre igual em todas as execuções
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 3. Construção da Arquitetura do Modelo
# ---------------------------------------------------------

# Criação do pipeline de Data Augmentation nativo do Keras
# Aplica transformações sintéticas aleatórias apenas durante o treinamento para evitar overfitting
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(32, 32, 3)), # Espelha lateralmente
    layers.RandomRotation(0.1),                               # Rotaciona em até 10%
    layers.RandomZoom(0.1),                                   # Aplica zoom de até 10%
], name = "data_augmentation_block")

# Construção da Rede Neural Convolucional (CNN)
model = models.Sequential([
    data_augmentation, # O bloco de augmentation atua como a primeira camada
    
    # Bloco Convolucional 1: Extração inicial de padrões básicos (bordas, cores)
    layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco Convolucional 2: Extração de características intermediárias (texturas, formas)
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    # Bloco Convolucional 3: Extração de padrões complexos (partes de objetos)
    layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    # Transição e Classificação Final
    layers.Flatten(),                       # Achata os mapas 2D em um vetor 1D
    layers.Dense(128, activation='relu'),   # Camada totalmente conectada (cérebro da rede)
    layers.Dropout(0.5),                    # Desliga 50% dos neurônios aleatoriamente p/ prevenir overfitting
    layers.Dense(10, activation='softmax')  # Saída com 10 classes e probabilidades (softmax)
])

# Imprime o resumo da arquitetura no terminal
model.summary()

# ---------------------------------------------------------
# 4. Compilação e Configuração de Treinamento
# ---------------------------------------------------------

# Otimizador Adam se adapta dinamicamente; Função de perda ideal para rótulos inteiros (sparse)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callback para interromper o treinamento se a validação parar de melhorar por 5 épocas seguidas
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True  
)

# ---------------------------------------------------------
# 5. Execução do Treinamento
# ---------------------------------------------------------

# O batch_size=16 previne o estouro de memória RAM, alimentando a rede em pequenos lotes
# Se houver maior poder de processamento, considere aumentar o batch_size (ex: 32 ou 64) para acelerar o treinamento
print("Iniciando o treinamento...")
historico = model.fit(
    x_train, y_train, epochs=30, batch_size=16, validation_data=(x_val, y_val), callbacks=[early_stopping] 
)

# ---------------------------------------------------------
# 6. Avaliação e Salvamento
# ---------------------------------------------------------

# Avalia a acurácia final utilizando os melhores pesos restaurados pelo EarlyStopping
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

print("\n" + "="*30)
print(f"Treinamento Concluído!")
print(f"Acurácia de Validação Final: {val_acc * 100:.2f}%")
print("="*30 + "\n")

# Salva a arquitetura, pesos e estado do otimizador no arquivo "Mestre"
model.save("model.h5")
print("Modelo salvo com sucesso no arquivo: 'model.h5'")