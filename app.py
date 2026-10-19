import streamlit as st
from PIL import Image
from ultralytics import YOLO
import google.generativeai as genai

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y DISEÑO
# ==========================================
st.set_page_config(page_title="Monitor de Fatiga", page_icon="🚛", layout="centered")

# Título atractivo (Req 4.4)
st.title("🚛 Monitor de Seguridad: Detección de Fatiga")
st.markdown("Sube una imagen de la cabina para analizar el estado de alerta del conductor.")

# Barra lateral con advertencias (Req 4.3 y 4.5)
with st.sidebar:
    st.header("⚠️ Advertencia Responsable")
    st.info("Esta es una herramienta de uso educativo y preventivo para seguridad industrial. **No constituye un diagnóstico médico**. Las predicciones generadas por Inteligencia Artificial deben ser siempre verificadas por un supervisor humano, especialmente si el nivel de confianza es bajo.")
    st.markdown("---")
    st.markdown("**Modelo de Visión:** YOLO26-cls")
    st.markdown("**IA Generativa:** Google Gemini")

# ==========================================
# 2. CARGAR EL MODELO DE VISIÓN (YOLO)
# ==========================================
# st.cache_resource hace que el modelo se cargue una sola vez y la página no se ponga lenta
@st.cache_resource
def cargar_modelo():
    # Aquí llamamos a tu archivo .pt que descargaste
    return YOLO("modelo_fatiga_best.pt")

modelo_vision = cargar_modelo()

# ==========================================
# 3. INTERFAZ DE CARGA DE IMAGEN (Req 4.3)
# ==========================================
archivo_subido = st.file_uploader("Sube la foto del operario aquí (JPG, PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is not None:
    # Mostramos la imagen que subió el usuario
    imagen = Image.open(archivo_subido)
    st.image(imagen, caption="Imagen cargada para análisis", use_container_width=True)

    # ==========================================
# 4. INFERENCIA: LA MÁQUINA MIRA LA FOTO
# ==========================================
    with st.spinner('Analizando imagen...'):
        # Le pasamos la foto al modelo YOLO
        resultados = modelo_vision(imagen)
        
        # Extraemos la clase ganadora y su porcentaje de confianza
        clase_predicha = resultados[0].names[resultados[0].probs.top1]
        confianza = float(resultados[0].probs.top1conf) * 100

    # Mostramos los resultados en cajitas bonitas
    st.subheader("📊 Resultados Técnicos")
    col1, col2 = st.columns(2)
    col1.metric("Diagnóstico Visual", clase_predicha.replace("_", " ").title())
    col2.metric("Nivel de Confianza", f"{confianza:.1f}%")

    # ==========================================
    # 5. IA GENERATIVA: EL ASESOR VIRTUAL (Req 4.5)
    # ==========================================
    st.markdown("---")
    st.subheader("💬 Asesor Virtual en Seguridad")
    
    if st.button("Generar Protocolo de Acción"):
        with st.spinner("Consultando al asesor de IA..."):
            try:
                # Conectamos con la llave secreta de Gemini que pondremos en GitHub
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                modelo_llm = genai.GenerativeModel('gemini-1.5-flash')

                # El Prompt estructurado y responsable
                prompt = f"""
                Eres un asesor experto en seguridad industrial y salud ocupacional.
                Un sistema de visión artificial acaba de analizar a un chofer de camión y detectó la clase: '{clase_predicha}' 
                con un nivel de confianza del {confianza:.1f}%.

                Instrucciones para tu respuesta:
                1. Si la confianza es menor al 80%, inicia tu respuesta con un gran aviso que diga: '⚠️ REVISIÓN HUMANA REQUERIDA'.
                2. Si la clase es 'ojos_cerrados', redacta un protocolo de emergencia urgente en viñetas para detener el vehículo, contactar al chofer por radio y aplicar pausa activa.
                3. Si la clase es 'ojos_abiertos', redacta un breve mensaje de validación indicando que el operario está en condiciones óptimas.
                4. Usa un tono ejecutivo, directo y fácil de entender.
                """

                # Generamos la respuesta
                respuesta = modelo_llm.generate_content(prompt)
                
                # Pintamos la respuesta de rojo si es peligro, o verde si está bien
                if clase_predicha == "ojos_cerrados":
                    st.error(respuesta.text)
                else:
                    st.success(respuesta.text)

            except Exception as e:
                st.error(f"¡Pillado! El error real de la máquina es: {e}")
