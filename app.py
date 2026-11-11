"""Interfaz del clasificador de sentimiento del ACA 2."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Cambia este valor después de publicar el modelo en tu cuenta de Hugging Face.
MODEL_REPO_ID = "Andrresgil/aca2-sentimiento-retail"


@st.cache_resource(show_spinner="Cargando el modelo de sentimiento...")
def cargar_modelo(repo_id: str):
    """Descarga una vez el modelo y el tokenizador; reutiliza ambos en cada consulta."""
    tokenizador = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    modelo = AutoModelForSequenceClassification.from_pretrained(repo_id)
    modelo.eval()
    ruta_config = hf_hub_download(repo_id=repo_id, filename="preprocesamiento.json")
    configuracion = json.loads(Path(ruta_config).read_text(encoding="utf-8"))
    return tokenizador, modelo, configuracion


def predecir(texto: str, tokenizador, modelo, max_length: int):
    """Devuelve la clase y las probabilidades usando el pipeline del cuaderno."""
    entradas = tokenizador(
        texto, return_tensors="pt", truncation=True, max_length=max_length
    )
    with torch.inference_mode():
        puntuaciones = modelo(**entradas).logits
        probabilidades = torch.softmax(puntuaciones, dim=-1)[0].tolist()
    etiquetas = [modelo.config.id2label[i] for i in range(modelo.config.num_labels)]
    indice = max(range(len(probabilidades)), key=probabilidades.__getitem__)
    return etiquetas[indice], dict(zip(etiquetas, probabilidades))


st.set_page_config(page_title="ACA 2 · Sentimiento de reseñas", page_icon="🛒")
st.title("🛒 Sentimiento de reseñas de productos")
st.write(
    "Escribe una reseña en **inglés**. El modelo DistilBERT estima si expresa "
    "una opinión negativa, neutra o positiva."
)

if not MODEL_REPO_ID or MODEL_REPO_ID.partition("/")[0].startswith("TU_"):
    st.info(
        "Antes de publicar la app, reemplaza TU_USUARIO en MODEL_REPO_ID "
        "por tu usuario y repositorio de Hugging Face."
    )
    st.stop()

try:
    tokenizador, modelo, configuracion = cargar_modelo(MODEL_REPO_ID)
except Exception as exc:
    st.error("No se pudo cargar el modelo. Revisa el identificador y los archivos en Hugging Face.")
    st.exception(exc)
    st.stop()

ejemplos = {
    "Escribir mi propia reseña": "",
    "Ejemplo positivo": "The tablet is easy to use and the battery lasts all day.",
    "Ejemplo neutro": "The product works, but the screen could be brighter.",
    "Ejemplo negativo": "The device stopped working after a week and I am disappointed.",
}
ejemplo = st.selectbox("Elige un ejemplo o escribe tu reseña", list(ejemplos))
texto = st.text_area(
    "Texto de la reseña",
    value=ejemplos[ejemplo],
    height=140,
    placeholder="Write a product review in English...",
)

if st.button("Analizar reseña", type="primary"):
    if not texto.strip():
        st.warning("Escribe una reseña antes de analizarla.")
    else:
        max_length = int(configuracion["max_length"])
        longitud = len(tokenizador(texto, truncation=False)["input_ids"])
        if longitud > max_length:
            st.info(
                f"La reseña tiene {longitud} tokens. El modelo usará los primeros "
                f"{max_length}, como en el cuaderno."
            )
        etiqueta, probabilidades = predecir(texto, tokenizador, modelo, max_length)
        st.subheader(f"Resultado: {etiqueta}")
        tabla = pd.DataFrame(
            {"Sentimiento": list(probabilidades),
             "Probabilidad": list(probabilidades.values())}
        )
        st.bar_chart(tabla, x="Sentimiento", y="Probabilidad", y_label="Probabilidad")
        st.caption(
            "Estas probabilidades expresan la confianza relativa del modelo; "
            "no garantizan que la clasificación sea correcta."
        )

with st.expander("Cómo interpretar este proyecto"):
    st.write(
        "En 9.098 reseñas de prueba, el modelo obtuvo 93,58 % de exactitud y "
        "F1 macro de 0,6720. Reconoce mejor las opiniones positivas; las neutras "
        "son las más difíciles (F1 = 0,3283). Las etiquetas de entrenamiento "
        "provienen de las calificaciones por estrellas y son aproximadas."
    )
