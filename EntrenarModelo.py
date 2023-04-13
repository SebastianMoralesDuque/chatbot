import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# Cargar datos del archivo JSON
with open("intents.json", encoding='utf-8') as archivo:
    datos = json.load(archivo)

# Preprocesamiento de los datos
entrenamiento = []
clases = []
documentos = []
ignorar = ["?", "!", ".", ","]

for intent in datos["intents"]:
    for patron in intent["patterns"]:
        # Convertir a minúsculas y eliminar signos de puntuación
        palabras = [palabra.lower() for palabra in patron.split() if palabra not in ignorar]
        entrenamiento.append(" ".join(palabras))
        clases.append(intent["tag"])
        documentos.append((palabras, intent["tag"]))

# Crear diccionario de palabras
tokenizer = Tokenizer()
tokenizer.fit_on_texts(entrenamiento)
palabras = tokenizer.word_index
num_palabras = len(palabras) + 1

# Crear datos de entrenamiento
entradas = []
salidas = []
for doc in documentos:
    # Crear vectores one-hot para la entrada
    entrada = [0] * num_palabras
    for palabra in doc[0]:
        if palabra in palabras:
            entrada[palabras[palabra]] = 1
    entradas.append(entrada)

    # Crear vectores one-hot para la salida
    salida = [0] * len(clases)
    salida[clases.index(doc[1])] = 1
    salidas.append(salida)

# Convertir a matrices numpy
X = np.array(entradas)
Y = np.array(salidas)

# Definir modelo de red neuronal
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_dim=num_palabras, activation='relu'),
    tf.keras.layers.Dense(len(clases), activation='softmax')
])
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entrenar modelo
modelo.fit(X, Y, epochs=1000, batch_size=64, verbose=1)

#guardar modelo
modelo.save('modelo_chatbot_final.h5')

