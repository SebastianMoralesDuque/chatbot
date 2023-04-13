import string
from tkinter import scrolledtext, ttk
import time
import numpy as np
import tensorflow as tf
import unicodedata
from tensorflow.keras.preprocessing.text import Tokenizer
import tkinter as tk
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

#cargar modelo
modelo = tf.keras.models.load_model('modelo_chatbot_final.h5')

def procesar_entrada(entrada):
    # Eliminar signos de puntuación y tildes
    entrada = entrada.lower()
    entrada = entrada.translate(str.maketrans('', '', string.punctuation))
    entrada = unicodedata.normalize('NFKD', entrada).encode('ASCII', 'ignore').decode('utf-8')
    return entrada

def chatbot_respuesta(texto):
    global response_time  # Permite el acceso a la variable global
    # Mide el tiempo de respuesta
    start_time = time.time()
    #procesa la entrada del usuario para quitar tildes y signos
    texto=procesar_entrada(texto)
    #verifica que el usuario no haya escrito de nuevo lo mismo
    chat_history_text = chat_history.get("1.0", tk.END)
    last_message_start = chat_history_text.rfind("Tú:")
    if last_message_start != -1:
        last_message_end = chat_history_text.find("\n\n", last_message_start)
        last_message = chat_history_text[last_message_start:last_message_end]
        last_message=procesar_entrada(last_message)
        if texto.lower() in last_message.lower():
            return "Ya has dicho eso antes. ¿Hay algo más en lo que pueda ayudarte?"

    # Preprocesamiento de la entrada
    entrada = [0] * num_palabras
    palabras_entrada = [palabra.lower() for palabra in texto.split() if palabra not in ignorar]
    for palabra in palabras_entrada:
        if palabra in palabras:
            entrada[palabras[palabra]] = 1

    # Predecir respuesta con modelo
    prediccion = modelo.predict(np.array([entrada]))
    respuesta_index = np.argmax(prediccion)
    tag_respuesta = clases[respuesta_index]

    # Asociar pregunta con respuesta
    preguntas_respuestas = {}
    for intent in datos["intents"]:
        if intent["tag"] == tag_respuesta:
            patterns = intent["patterns"]
            responses = intent["responses"]
            if len(patterns) == len(responses):
              for i, pattern in enumerate(patterns):
                pregunta = procesar_entrada(pattern)
                respuesta = responses[i]
                preguntas_respuestas[pregunta] = respuesta

    # Seleccionar respuesta aleatoria
    if preguntas_respuestas:
        respuesta = preguntas_respuestas.get(texto.lower(), "")
        if respuesta == "":
            respuesta = np.random.choice(list(preguntas_respuestas.values()))
    else:
        respuesta = np.random.choice(responses)

    end_time = time.time()
    response_time = end_time - start_time
    response_time = str(format(response_time, '.3f'))
    return respuesta

# Crea la interfaz gráfica del chatbot
def send():
    # Obtiene la entrada del usuario
    user_input = input_box.get()
    # Limpia la entrada del usuario
    input_box.delete(0, tk.END)
    # Clasifica la entrada del usuario y obtiene la respuesta del chatbot
    bot_response = chatbot_respuesta(user_input)
    # Agrega la entrada del usuario al historial de chat
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, 'Tú: ' + (user_input) + '\n\n')
    chat_history.see(tk.END)
    chat_history.config(state=tk.DISABLED)
    # Quitar placeholder
    input_box.delete(0, tk.END)
    input_box.insert(0, '')
    input_box.configure(foreground='black')

    def show_bot_response(response, index=0):
        if index < len(response):
            chat_history.config(state=tk.NORMAL)
            if index == 0:
                chat_history.insert(tk.END, 'Sara: ')
            chat_history.insert(tk.END, response[index])
            if index == len(response) - 1:
                chat_history.insert(tk.END, '\n\n')
                input_box.config(state=tk.NORMAL)  # Habilita la entrada del usuario
                # Actualiza el tiempo de respuesta en el Label
                response_time_label.config(text="Tiempo de respuesta: " + response_time)
            chat_history.config(state=tk.DISABLED)
            chat_history.yview(tk.END)
            chat_history.update()
            root.after(10, show_bot_response, response, index + 1)
    root.after(1000, show_bot_response, bot_response, 0)
    input_box.config(state=tk.DISABLED)  # Deshabilita la entrada del usuario

def on_entry_click(event):
    if input_box.get() == 'Escriba su mensaje aquí':
        input_box.delete(0, tk.END)
        input_box.configure(foreground='black')

root = tk.Tk()
root.title("Chatbot")
root.geometry("800x600")
root.resizable(width=False, height=False)

# Agregamos un título y un mensaje de bienvenida
title_label = tk.Label(root, text="¡Bienvenido al Chatbot!", font=("Arial", 20, "bold"))
title_label.grid(column=1, row=0, padx=10, pady=10)
welcome_message = "Hola, ¿en qué puedo ayudarte?"

# Ajustamos el historial del chat
chat_history = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=50, height=20)
chat_history.config(state=tk.DISABLED)
chat_history.configure(font=('Arial', 12), foreground='white', background='black', insertbackground='white')
chat_history.insert(tk.END, 'Chatbot: ' + welcome_message + '\n\n')
chat_history.grid(column=1, row=1, padx=10, pady=10)
chat_history.focus()

# estilo para el placeholder
style = ttk.Style(root)
style.configure('Placeholder.TEntry', foreground='grey')

# Ajustamos el tamaño y el padding de la caja de entrada y el botón
input_box = ttk.Entry(root, width=50)
input_box.configure(font=('Arial', 12), foreground='black', background='white')
input_box.insert(0, 'Escriba su mensaje aquí')
input_box.bind("<Return>", (lambda event: send()))
input_box.bind('<FocusIn>', on_entry_click)  # Agrega el evento de clic
input_box.grid(column=1, row=2, padx=10, pady=10, sticky="w")
send_button = tk.Button(root, text="Enviar", command=send, width=10)
send_button.configure(font=('Arial', 12))
send_button.grid(column=1, row=2, padx=10, pady=10, sticky="e")

# Ajustamos la posición de la etiqueta de tiempo de respuesta
response_time_label = tk.Label(root, text="", font=("Arial", 8))
response_time_label.grid(column=1, row=2, padx=180, pady=10, sticky="e")

# Centramos el chat_history y agregamos un padding alrededor
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(1, weight=1)
chat_history.grid(sticky="nsew", padx=10, pady=10)

# Ajustamos las opciones de fuente para todos los elementos
root.option_add('*Font', 'Arial Unicode MS 12')
root.option_add('*Button.Font', 'Arial Unicode MS 12')
root.option_add('*Entry.Font', 'Arial Unicode MS 12')
root.option_add('*Label.Font', 'Arial Unicode MS 12')
root.option_add('*Message.Font', 'Arial Unicode MS 12')
root.option_add('*Text.Font', 'Arial Unicode MS 12')

# Ajustamos el tamaño de fuente del chat_history
chat_history.config(font=("Arial Unicode MS", 12))

root.mainloop()