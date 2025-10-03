# Detecção de Objetos em Tempo Real com YOLO e Análise de Tons de Pele

![projeto](https://github.com/user-attachments/assets/8a70aa9e-e2bc-45c2-a28a-16db7ca827bd)


Este projeto implementa um sistema de detecção de objetos em tempo real usando YOLOv8, com funcionalidade adicional para análise de tons de pele em pessoas detectadas.

## 📋 Descrição

O código utiliza a webcam para capturar vídeo em tempo real e aplica o modelo YOLOv8 para detectar objetos. Para cada pessoa detectada, o sistema realiza uma análise de cor predominante na região para estimar características de tom de pele no espaço de cores HSV.

## ✨ Funcionalidades

- **Detecção de objetos em tempo real** usando YOLOv8
- **Identificação de múltiplas classes** (pessoas, carros, etc.)
- **Cálculo da área relativa** de cada objeto detectado
- **Análise de tons de pele** para pessoas detectadas
- **Interface visual** com bounding boxes e informações detalhadas

## 🛠️ Tecnologias Utilizadas

- **OpenCV** - Processamento de imagem e vídeo
- **Ultralytics YOLOv8** - Detecção de objetos
- **NumPy** - Cálculos numéricos e manipulação de arrays

## 📦 Instalação

### Pré-requisitos
- Python 3.7+
- Webcam funcionando

### Dependências
```bash
pip install opencv-python ultralytics numpy
