# 🚛 Monitor de Seguridad : Detección de Fatiga

## 1. Descripción del Proyecto
Aplicación web diseñada para la inspección visual en sistemas de autolavado y flotas. Permite detectar si un operario o conductor presenta fatiga (ojos cerrados) o se encuentra alerta (ojos abiertos) con el fin de prevenir accidentes laborales. Al detectar peligro, un asistente de IA Generativa redacta un protocolo de emergencia.

## 2. Datos Utilizados
- **Origen:** Driver Drowsiness Dataset (DDD) obtenido desde Kaggle.
- **Licencia:** Pública / Abierta.
- **Clases:** 2 clases (`ojos_abiertos`, `ojos_cerrados`).
- **Balance y división:** Se utilizaron 100 imágenes por clase (200 en total). El conjunto está balanceado y dividido en 80 imágenes para entrenamiento (Train) y 20 imágenes para validación (Val) por cada clase.
- **Limitaciones:** El modelo fue entrenado con recortes claros de ojos. En una cabina real, factores como iluminación deficiente, vibraciones o lentes de sol podrían generar un sesgo en la predicción.

## 3. Modelo Elegido
Se seleccionó **YOLO26-cls** (o YOLO11-cls).
- **Justificación:** Se comparó inicialmente con un modelo Zero-Shot (CLIP de Hugging Face), el cual obtuvo un accuracy cercano al azar (52.5%). YOLO, mediante *transfer learning*, logró un accuracy de 92.5% en pocas épocas, demostrando ser mucho más eficiente, liviano y apto para poner en producción.
- **Análisis de errores:** Se identificó que el error más costoso (Falso Negativo) es predecir ojos abiertos cuando están cerrados. Por esto, si el modelo tiene una confianza menor al 80%, el sistema exige una revisión humana obligatoria.

## 4. Instrucciones de Uso (Local)
Para correr este proyecto en tu computador:
1. Instalar las librerías: `pip install -r requirements.txt`
2. Ejecutar la app: `streamlit run app.py`

## 5. Integrantes
- Felipe Antonio Lillo
- Ja Ming Louie Lei

## 6. Declaración de uso de IA
Para el desarrollo de este proyecto se utilizó IA (Gemini/ChatGPT) como apoyo.
- **Uso:** Se utilizó para estructurar el código base de Streamlit y configurar la lógica del entrenamiento en Colab.
- **Adaptación:** El código fue adaptado y extendido específicamente al problema propuesto, ajustando el *prompt* de seguridad industrial y los umbrales de decisión según la matriz de confusión generada.
