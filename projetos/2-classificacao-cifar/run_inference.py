"""
Módulo de Inferência - Projeto CIFAR-10 Edge
Este script carrega o modelo otimizado (model.tflite) e realiza inferências
locais utilizando imagens do conjunto de testes do CIFAR-10.
Objetivo: Validar a acurácia e o funcionamento do modelo convertido para Edge.
"""

import os
import sys
import io
import numpy as np
import tensorflow as tf

# ---------------------------------------------------------
# Configuração de Ambiente
# ---------------------------------------------------------

# Força o stdout e stderr a usar UTF-8 para evitar erros de 
# codificação (UnicodeEncodeError) ao imprimir no terminal do Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ---------------------------------------------------------
# Constantes do Projeto
# ---------------------------------------------------------

# Número de amostras que serão testadas durante a execução
N_SAMPLES = 5

# Rótulos (classes) oficiais do dataset CIFAR-10 na ordem correta
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def main():
    """
    Função principal que gerencia o fluxo de inferência:
    1. Resolve os caminhos e carrega o interpretador TFLite.
    2. Baixa e pré-processa as imagens de teste.
    3. Executa as predições e exibe o comparativo no terminal.
    """

    # 1. Configuração do Modelo TFLite
    # Obtém o diretório absoluto do script atual para evitar erros de "File Not Found"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")

    # Instancia o interpretador de borda (Edge) e aloca memória para os tensores
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Mapeia as propriedades da camada de entrada e da camada de saída do modelo
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Carregamento e Pré-processamento dos Dados
    # Carrega o CIFAR-10, ignorando o conjunto de treino (_, _) pois usaremos apenas o teste
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normaliza os pixels das imagens para o intervalo [0, 1] (mesmo padrão do treinamento)
    x_test = x_test.astype("float32") / 255.0

    # Achata o array de rótulos (labels) para facilitar a leitura no formato de lista simples
    y_test = y_test.reshape(-1)

    # 3. Loop de Inferência
    print(f"Rodando inferência em {N_SAMPLES} amostras usando model.tflite:\n")
    for i in range(N_SAMPLES):
        # Pega uma única imagem, adiciona a dimensão de lote (batch=1) exigida pelo modelo
        # e garante que o tipo de dado bate com a entrada esperada (float32 ou int8)
        sample = np.expand_dims(x_test[i], axis=0).astype(input_details[0]["dtype"])

        # Alimenta a imagem no interpretador
        interpreter.set_tensor(input_details[0]["index"], sample)

        # Executa a rede neural (inferência)
        interpreter.invoke()

        # Extrai o array com as probabilidades preditas para as 10 classes
        pred = interpreter.get_tensor(output_details[0]["index"])[0]

        # Encontra o índice da classe com a maior probabilidade (np.argmax)
        predicted_class = int(np.argmax(pred))

        # Imprime o resultado final traduzindo os índices para os nomes das classes
        print(
            f"Amostra {i + 1}: predito={CLASS_NAMES[predicted_class]} "
            f"| real={CLASS_NAMES[int(y_test[i])]}"
        )

# Garante que a função main() só seja executada se este arquivo for rodado diretamente
if __name__ == "__main__":
    main()
