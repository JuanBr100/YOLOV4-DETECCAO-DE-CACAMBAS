import pkgutil
import os

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox


model_cfg = 'yolov4_custom.cfg'
model_weights = 'yolov4_custom_3000.weights'


net = cv2.dnn.readNet(model_weights, model_cfg)


classes = ["Cacamba"]

diretorio_entrada = None
diretorio_saida = None

def show_completion_dialog():
    completion_dialog = tk.Tk()
    completion_dialog.title("Processo Concluído")

    completion_label = tk.Label(completion_dialog, text="Processo de geração de imagens concluído.")
    completion_label.pack(padx=20, pady=10)

    ok_button = tk.Button(completion_dialog, text="OK", command=completion_dialog.quit)
    ok_button.pack(pady=10)

    completion_dialog.mainloop()
    completion_dialog.destroy()

    root.quit()

# Função para selecionar um diretório de imagens de entrada
def selec_diretorio_entrada():
    global diretorio_entrada
    diretorio_entrada = filedialog.askdirectory()


# Função para selecionar um diretório de saída
def selec_diretorio_saida():
    global diretorio_saida
    diretorio_saida = filedialog.askdirectory()


# Função para iniciar a detecção nas imagens do diretório de entrada
def detectar_diretorio():
    if not diretorio_entrada:
        messagebox.showerror("Erro", "Nenhum diretório de entrada selecionado.")
        return
    if not diretorio_saida:
        messagebox.showerror("Erro", "Nenhum diretório de saída selecionado.")
        return

    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)

    for root, dirs, files in os.walk(diretorio_entrada):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                if image is not None:  # Verifique se a imagem é válida
                    detected_image = detect_objects(image)
                    output_path = os.path.join(diretorio_saida, file)
                    cv2.imwrite(output_path, detected_image)
                else:
                    print(f"Erro ao ler a imagem: {image_path}")

    cv2.destroyAllWindows()

    show_completion_dialog()


# Função para detecção de objetos e desenho de retângulos
def detect_objects(frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outs = net.forward(layer_names)

    class_ids = []
    confidences = []
    boxes = []
    width, height = frame.shape[1], frame.shape[0]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Filtrar predições com confiança acima de 70%
    indexes = [i for i in indexes if confidences[i] > 0.70]

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            rectangle_color = (0, 0, 255)  # Cor do retângulo (azul, verde, vermelho)
            cv2.rectangle(frame, (x, y), (x + w, y + h), rectangle_color, 2)

            text = f"{label} {confidence:.2f}"
            text_color = (0, 0, 255)  # Cor do texto (vermelho)
            font_scale = 0.5
            font = cv2.FONT_HERSHEY_SIMPLEX

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 2)

            text_x = x
            text_y = y - 10

            cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, 2)

    return frame

root = tk.Tk()
root.title("Detecção de Objetos em Diretório")

select_entrada_button = tk.Button(root, text="Selecionar Diretório de Entrada", command=selec_diretorio_entrada)
select_entrada_button.pack()

select_saida_button = tk.Button(root, text="Selecionar Diretório de Saída", command=selec_diretorio_saida)
select_saida_button.pack()

start_button = tk.Button(root, text="Iniciar Detecção no Diretório", command=detectar_diretorio)
start_button.pack()

root.mainloop()
