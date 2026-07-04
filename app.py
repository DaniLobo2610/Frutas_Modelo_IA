# ==================================================
# IMPORTACIONES
# ==================================================

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image

# ==================================================
# CONFIGURACIÓN
# ==================================================

st.set_page_config(

    page_title="Modelo Predictivo de Frutas",

    page_icon="🍎",

    layout="wide"

)


# ==================================================
# ESTILOS
# ==================================================

st.markdown("""

<style>

.main{

    background-color:#0E1117;

}

.titulo{

    background:linear-gradient(90deg,#FF512F,#F09819);

    padding:25px;

    border-radius:20px;

    text-align:center;

    color:white;

    box-shadow:0px 8px 18px rgba(0,0,0,.35);

}

.info{

    background:#1E293B;

    padding:20px;

    border-radius:15px;

    color:white;

    margin-top:20px;

    margin-bottom:20px;

}

.frutas{

    background:#172033;

    padding:20px;

    border-radius:15px;

    color:white;

}

.footer{

    text-align:center;

    color:gray;

    margin-top:30px;

}

</style>

""", unsafe_allow_html=True)


# ==================================================
# ENCABEZADO
# ==================================================

st.markdown("""

<div class="titulo">

<h1>🍎🍌🍓 Modelo Predictivo de Frutas</h1>

<h3>Universidad Nacional Autónoma de Honduras</h3>

<h3>Ingeniería en Sistemas</h3>

<h3>Inteligencia Artificial</h3>

</div>

""", unsafe_allow_html=True)

st.markdown("""

<div class="info">

<b>👨‍💻 Desarrollado por:</b>

<br>

Alberto Daniel Lobo Chavarría

<br><br>

Este sistema utiliza un modelo de Deep Learning basado en
<b>MobileNetV2</b> para reconocer automáticamente diferentes tipos
de frutas a partir de una imagen.

</div>

""", unsafe_allow_html=True)


# ==================================================
# RUTAS
# ==================================================

IMG_SIZE = (224,224)

MODEL_DIR = Path("Modelo_frutas")

MODEL_PATH = MODEL_DIR / "modelo_frutas.keras"

CLASS_PATH = MODEL_DIR / "class_names.json"


# ==================================================
# CARGAR MODELO
# ==================================================

@st.cache_resource

def cargar_modelo():

    return tf.keras.models.load_model(

        MODEL_PATH,

        compile=False

    )


@st.cache_data

def cargar_clases():

    with open(CLASS_PATH,"r",encoding="utf-8") as f:

        return json.load(f)

modelo = cargar_modelo()

clases = cargar_clases()


# ==================================================
# Frutas Disponibles
# ==================================================

EMOJIS = {

    "Apple": "🍎 Apple - Manzana",

    "Avocado": "🥑 Avocado - Aguacate",

    "Banana": "🍌 Banana - Banano",

    "Blackberry": "🫐 Blackberry - Mora",

    "Carambola": "⭐ Carambola - Carambola",

    "Cherry": "🍒 Cherry - Cereza",

    "Coconut": "🥥 Coconut - Coco",

    "Grape": "🍇 Grape - Uva",

    "Orange": "🍊 Orange - Naranja",

    "Papaya": "🥭 Papaya - Papaya",

    "Peach": "🍑 Peach - Durazno",

    "Pear": "🍐 Pear - Pera",

    "Raspberry": "🫐 Raspberry - Frambuesa",

    "Strawberry": "🍓 Strawberry - Fresa",

    "Tomato": "🍅 Tomato - Tomate"

}


# ==================================================
# FRUTAS DISPONIBLES
# ==================================================

st.markdown("""

<div class="frutas">

<h3>🍉 Frutas que reconoce el modelo</h3>

</div>

""", unsafe_allow_html=True)

col1,col2,col3,col4,col5 = st.columns(5)

lista = list(EMOJIS.items())

for i,(nombre,emoji) in enumerate(lista):

    if i%5==0:
        col1.write(f"{emoji} {nombre}")

    elif i%5==1:
        col2.write(f"{emoji} {nombre}")

    elif i%5==2:
        col3.write(f"{emoji} {nombre}")

    elif i%5==3:
        col4.write(f"{emoji} {nombre}")

    else:
        col5.write(f"{emoji} {nombre}")

# ==================================================
# PREPARAR IMAGEN
# ==================================================

def preparar_imagen(imagen):

    imagen = imagen.convert("RGB")

    imagen = imagen.resize(IMG_SIZE)

    img = np.array(imagen, dtype=np.float32)

    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


