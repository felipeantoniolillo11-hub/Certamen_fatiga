# =========================================================================================
# LIBRERÍAS (Las herramientas que necesitamos para trabajar)
# =========================================================================================
import streamlit as st               # La librería mágica que convierte este código en una página web.
from PIL import Image                # Librería para poder abrir, leer y mostrar las fotos que suba el usuario.
from ultralytics import YOLO         # La herramienta de visión artificial para usar nuestro modelo entrenado.
import google.generativeai as genai  # La librería oficial de Google para conectar con el cerebro de Gemini.

# =========================================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y DISEÑO (La estructura visual de nuestra app)
# =========================================================================================
# Aquí definimos cómo se ve la pestaña del navegador: el nombre, el emoji y que el diseño quede centrado.
st.set_page_config(page_title="Monitor de Fatiga", page_icon="🚛", layout="centered")

# 'st.title' crea el título principal en letra grande y negrita.
st.title("🚛 Monitor de Seguridad: Detección de Fatiga")
# 'st.markdown' escribe el texto de instrucciones justo debajo del título.
st.markdown("Sube una imagen de la cabina para analizar el estado de alerta del conductor.")

# =========================================================================================
# 1.1 BARRA LATERAL (El menú de la izquierda)
# =========================================================================================
# 'with st.sidebar:' indica que todo lo que esté aquí adentro se mostrará en el menú lateral.
with st.sidebar:
    # Creamos un título pequeño para la sección de advertencias.
    st.header("⚠️ Advertencia Responsable")
    # 'st.info' crea una cajita azul para mostrar un mensaje importante de ética y seguridad.
    st.info("Esta es una herramienta de uso educativo y preventivo para seguridad industrial. **No constituye un diagnóstico médico**. Las predicciones generadas por Inteligencia Artificial deben ser siempre verificadas por un supervisor humano, especialmente si el nivel de confianza es bajo.")
    # 'st.markdown("---")' dibuja una línea horizontal separadora.
    st.markdown("---")
    # Agregamos texto en negrita (**) para indicar qué tecnologías estamos usando.
    st.markdown("**Modelo de Visión:** YOLO26-cls")
    st.markdown("**IA Generativa:** Google Gemini")

# =========================================================================================
# 2. CARGAR EL MODELO DE VISIÓN YOLO (El "Ojo" del sistema)
# =========================================================================================
# '@st.cache_resource' es súper importante: hace que el modelo pesado se cargue SOLO UNA VEZ.
# Si no ponemos esto, la página web se congelaría cada vez que el usuario haga clic en un botón.
@st.cache_resource
def cargar_modelo():
    # Buscamos y cargamos el archivo de pesos (.pt) que entrenamos previamente en Google Colab.
    return YOLO("modelo_fatiga_best.pt")

# Ejecutamos la función anterior y guardamos el modelo listo para usar en la variable 'modelo_vision'.
modelo_vision = cargar_modelo()

# =========================================================================================
# 3. INTERFAZ DE CARGA DE IMAGEN (Donde el usuario sube la foto)
# =========================================================================================
# 'st.file_uploader' crea el cajón donde el usuario arrastra su foto. Limitamos los formatos a JPG y PNG.
archivo_subido = st.file_uploader("Sube la foto del operario aquí (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Si la variable no está vacía (es decir, si el usuario efectivamente subió una foto), ejecutamos el análisis:
if archivo_subido is not None:
    # Usamos la librería PIL (Image) para abrir el archivo subido en la memoria.
    imagen = Image.open(archivo_subido)
    # 'st.image' muestra la foto en grande en la página web para que el usuario la vea.
    st.image(imagen, caption="Imagen cargada para análisis", use_container_width=True)

# =========================================================================================
# 4. INFERENCIA: LA MÁQUINA MIRA LA FOTO (Análisis de YOLO)
# =========================================================================================
    # 'st.spinner' pone un círculo girando en la pantalla con el texto "Analizando imagen..." mientras procesa.
    with st.spinner('Analizando imagen...'):
        # Le pasamos la foto cargada al cerebro de YOLO para que nos devuelva su predicción.
        resultados = modelo_vision(imagen)
        
        # 'resultados[0].probs.top1' saca el número de la clase que ganó (0 para abiertos, 1 para cerrados).
        # 'resultados[0].names' busca el texto asociado a ese número ("ojos_abiertos" u "ojos_cerrados").
        clase_predicha = resultados[0].names[resultados[0].probs.top1]
        # 'top1conf' saca el nivel de seguridad (ej. 0.96). Lo multiplicamos por 100 para que sea un porcentaje (96%).
        confianza = float(resultados[0].probs.top1conf) * 100

    # Creamos un subtítulo para mostrar los números que sacó YOLO.
    st.subheader("📊 Resultados Técnicos")
    # 'st.columns(2)' divide la pantalla en 2 mitades iguales.
    col1, col2 = st.columns(2)
    # En la columna 1, 'metric' crea un número grande. Limpiamos el texto (sacamos el guion bajo y ponemos mayúsculas).
    col1.metric("Diagnóstico Visual", clase_predicha.replace("_", " ").title())
    # En la columna 2, mostramos el porcentaje formateado a 1 decimal (ej. 99.8%).
    col2.metric("Nivel de Confianza", f"{confianza:.1f}%")

# =========================================================================================
# 5. IA GENERATIVA: EL ASESOR VIRTUAL (El "Cerebro" de Google Gemini)
# =========================================================================================
    # Separador visual.
    st.markdown("---")
    st.subheader("💬 Asesor Virtual en Seguridad")
    
    # Creamos un botón. Si el usuario lo presiona, el código de abajo comienza a correr.
    if st.button("Generar Protocolo de Acción"):
        # Mostramos la ruedita girando mientras nos conectamos a internet.
        with st.spinner("Consultando al asesor de IA..."):
            
            # El 'try' es la red de seguridad: "Intenta hacer esto, pero si falla, no destruyas la página".
            try:
                # Nos identificamos en Google usando la clave guardada en los Secretos de Streamlit Cloud.
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                
                # Cargamos la versión específica del modelo de lenguaje que recomendó la API.
                modelo_llm = genai.GenerativeModel('models/gemini-3.8-flash')
                
                # =========================================================================================
                # 5.1 INGENIERÍA DE PROMPTS (Las instrucciones para la IA)
                # =========================================================================================
                # Creamos el texto dinámico que le mandaremos a Gemini, inyectándole la clase y la confianza de YOLO.
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

                # Le mandamos el texto de arriba a Gemini y guardamos lo que nos responde.
                respuesta = modelo_llm.generate_content(prompt)
                
                # =========================================================================================
                # 5.2 PRESENTACIÓN DE RESULTADOS
                # =========================================================================================
                # Si YOLO detectó peligro (ojos cerrados), mostramos la respuesta de Gemini en una caja roja (st.error).
                if clase_predicha == "ojos_cerrados":
                    st.error(respuesta.text)
                # Si YOLO detectó que todo está bien, mostramos la respuesta en una caja verde (st.success).
                else:
                    st.success(respuesta.text)

            # Si algo de arriba falló (ej. se cayó el internet o la clave está mala), 'except' ataja el error.
            except Exception as e:
                # Mostramos un cartel rojo con el error técnico exacto para poder arreglarlo.
                st.error(f"¡Pillado! El error real de la máquina es: {e}")
