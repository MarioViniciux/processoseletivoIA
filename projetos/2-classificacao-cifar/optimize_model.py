"""
Módulo de Otimização - Projeto CIFAR-10 Edge
Este script carrega o modelo Keras original treinado (.h5) e o converte para 
o formato TensorFlow Lite (.tflite), otimizado para dispositivos de borda (Edge).
Aplica a técnica de Quantização de Faixa Dinâmica para reduzir significativamente 
o tamanho do arquivo, descartando o otimizador e comprimindo os pesos.
"""

import tensorflow as tf
import os

print("Iniciando a conversão e otimização do modelo...")

# ---------------------------------------------------------
# 1. Resolução Dinâmica de Caminhos
# ---------------------------------------------------------

# Obtém o diretório absoluto do script atual para evitar erros de leitura/gravação 
# caso o terminal seja aberto em um diretório diferente
script_dir = os.path.dirname(os.path.abspath(__file__))

# Constrói os caminhos absolutos seguros para os arquivos de entrada (.h5) e saída (.tflite)
model_h5_path = os.path.join(script_dir, "model.h5")
model_tflite_path = os.path.join(script_dir, "model.tflite")

# ---------------------------------------------------------
# 2. Carregamento do Modelo Original
# ---------------------------------------------------------

# O tf.keras.models.load_model reconstrói toda a arquitetura, os pesos e o 
# estado do otimizador a partir do arquivo Mestre salvo durante o treinamento
modelo = tf.keras.models.load_model(model_h5_path)
print(f"Modelo carregado com sucesso de: {model_h5_path}")

# ---------------------------------------------------------
# 3. Configuração da Conversão e Otimização
# ---------------------------------------------------------

# Inicializa o conversor do TensorFlow Lite extraindo a estrutura do modelo Keras
converter = tf.lite.TFLiteConverter.from_keras_model(modelo)

# Aplica a Quantização de Faixa Dinâmica (Dynamic Range Quantization)
# Esta etapa é crucial: converte os pesos de Float32 para Int8, garantindo 
# a redução de tamanho necessária para rodar o modelo em dispositivos Edge
converter.optimizations = [tf.lite.Optimize.DEFAULT]
print("Otimização configurada: Dynamic Range Quantization.")

# ---------------------------------------------------------
# 4. Execução da Conversão e Salvamento
# ---------------------------------------------------------

# Realiza o processo de conversão de fato, gerando o artefato final
tflite_model = converter.convert()

# Como o modelo compilado resultante é um arquivo binário (não é texto), 
# precisamos usar o modo "wb" (write binary) para salvá-lo no disco
with open(model_tflite_path, "wb") as f:
  f.write(tflite_model)

print(f"SUCESSO! Modelo otimizado e salvo em: {model_tflite_path}")