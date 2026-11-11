"""Interfaz del clasificador de sentimiento del ACA 2."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_REPO_ID = "Andrresgil/aca2-sentimiento-retail"
TRANSLATION_REPO_ID = "Helsinki-NLP/opus-mt-es-en"


@st.cache_resource(show_spinner="Cargando el modelo de sentimiento...")
def cargar_modelo(repo_id: str):
    tokenizador = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
    modelo = AutoModelForSequenceClassification.from_pretrained(repo_id)
    modelo.eval()
    ruta_config = hf_hub_download(repo_id=repo_id, filename="preprocesamiento.json")
    configuracion = json.loads(Path(ruta_config).read_text(encoding="utf-8"))
    return tokenizador, modelo, configuracion


@st.cache_resource(show_spinner="Cargando el traductor español → inglés...")
def cargar_traductor():
    tokenizador = AutoTokenizer.from_pretrained(TRANSLATION_REPO_ID)
    modelo = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_REPO_ID)
    modelo.eval()
    return tokenizador, modelo


def traducir_al_ingles(texto: str, tokenizador, modelo):
    longitud = len(tokenizador(texto, truncation=False)["input_ids"])
    entradas = tokenizador(
        texto, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.inference_mode():
        salida = modelo.generate(**entradas, max_new_tokens=256)
    traduccion = tokenizador.decode(salida[0], skip_special_tokens=True)
    return traduccion, longitud > 512


def predecir(texto: str, tokenizador, modelo, max_length: int):
    entradas = tokenizador(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        return_token_type_ids=False,
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
    "Escribe una reseña en inglés o español. DistilBERT estima si expresa "
    "una opinión negativa, neutra o positiva. Las reseñas en español se traducen "
    "al inglés antes del análisis."
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

idioma = st.radio("Idioma de la reseña", ["Español", "Inglés"], horizontal=True)
if idioma == "Español":
    ejemplos = {
        "Escribir mi propia reseña": "",
        "Ejemplo positivo": "La tableta es fácil de usar y la batería dura todo el día.",
        "Ejemplo neutro": "El producto funciona, pero la pantalla podría ser más brillante.",
        "Ejemplo negativo": "El dispositivo dejó de funcionar después de una semana y estoy decepcionado.",
    }
    texto_guia = "Escribe aquí una reseña en español..."
else:
    ejemplos = {
        "Escribir mi propia reseña": "",
        "Ejemplo positivo": "The tablet is easy to use and the battery lasts all day.",
        "Ejemplo neutro": "The product works, but the screen could be brighter.",
        "Ejemplo negativo": "The device stopped working after a week and I am disappointed.",
    }
    texto_guia = "Write a product review in English..."
ejemplo = st.selectbox("Elige un ejemplo o escribe tu reseña", list(ejemplos))
texto = st.text_area(
    "Texto de la reseña",
    value=ejemplos[ejemplo],
    height=140,
    placeholder=texto_guia,
    key=f"resena_{idioma}_{ejemplo}",
)

if st.button("Analizar reseña", type="primary"):
    if not texto.strip():
        st.warning("Escribe una reseña antes de analizarla.")
    else:
        texto_modelo = texto.strip()
        if idioma == "Español":
            try:
                tokenizador_traduccion, modelo_traduccion = cargar_traductor()
                texto_modelo, truncado = traducir_al_ingles(
                    texto_modelo, tokenizador_traduccion, modelo_traduccion
                )
            except Exception as exc:
                st.error("No se pudo traducir la reseña. Inténtalo de nuevo.")
                st.exception(exc)
                st.stop()
            if truncado:
                st.warning(
                    "La reseña supera el límite del traductor; solo se tradujo el comienzo."
                )
            st.write("**Traducción que recibió el clasificador:**", texto_modelo)
            st.caption(
                "La traducción puede cambiar matices de la opinión; "
                "esta ruta en español aún no tiene una evaluación cuantitativa propia."
            )

        max_length = int(configuracion["max_length"])
        longitud = len(tokenizador(texto_modelo, truncation=False)["input_ids"])
        if longitud > max_length:
            st.info(
                f"El texto en inglés tiene {longitud} tokens. El clasificador usará "
                f"los primeros {max_length}, como en el cuaderno."
            )
        etiqueta, probabilidades = predecir(
            texto_modelo, tokenizador, modelo, max_length
        )
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
        "En 9.098 reseñas de prueba en inglés, el modelo obtuvo 93,58 % de exactitud y "
        "F1 macro de 0,6720. Reconoce mejor las opiniones positivas; las neutras "
        "son las más difíciles (F1 = 0,3283). Las etiquetas de entrenamiento "
        "provienen de las calificaciones por estrellas y son aproximadas. "
        "Estas métricas no miden el rendimiento de la ruta de traducción desde español."
    )
