"""Interfaz del clasificador de sentimiento del ACA 2."""

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
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


def mostrar_recorrido_animado(etiqueta=None, probabilidades=None, tokens_usados=0):
    """Dibuja un esquema animado; no representa activaciones internas reales."""
    activo = etiqueta is not None and probabilidades is not None
    probabilidades = probabilidades or {}
    colores = {"Negativo": "#fb7185", "Neutro": "#fbbf24", "Positivo": "#4ade80"}
    color = colores.get(etiqueta, "#67e8f9")
    posiciones_x = [85, 205, 315, 425, 535, 645, 755, 855]
    alturas = [[150, 195, 240]] + [[125, 170, 215, 260]] * 6 + [[130, 190, 250]]
    etiquetas_salida = ["Negativo", "Neutro", "Positivo"]
    enlaces = []
    neuronas = []

    for paso in range(len(posiciones_x) - 1):
        for origen, y1 in enumerate(alturas[paso]):
            for destino, y2 in enumerate(alturas[paso + 1]):
                if abs(origen - destino) <= 1:
                    enlaces.append(
                        f'<line class="enlace" x1="{posiciones_x[paso]}" y1="{y1}" '
                        f'x2="{posiciones_x[paso + 1]}" y2="{y2}" '
                        f'style="animation-delay:{paso * 0.28:.2f}s" />'
                    )

    for paso, (x, columna) in enumerate(zip(posiciones_x, alturas)):
        for indice, y in enumerate(columna):
            es_resultado = activo and paso == len(posiciones_x) - 1 and etiquetas_salida[indice] == etiqueta
            clase = "neurona resultado" if es_resultado else "neurona"
            neuronas.append(
                f'<circle class="{clase}" cx="{x}" cy="{y}" r="{13 if es_resultado else 9}" '
                f'style="animation-delay:{paso * 0.28:.2f}s" />'
            )

    salidas = []
    for indice, nombre in enumerate(etiquetas_salida):
        valor = probabilidades.get(nombre, 0.0) * 100
        clase = "salida elegida" if nombre == etiqueta else "salida"
        texto_salida = f"{nombre} {valor:.1f} %" if activo else nombre
        salidas.append(
            f'<text class="{clase}" x="880" y="{alturas[-1][indice] + 5}">'
            f'{escape(texto_salida)}</text>'
        )

    ruta_activa = [(posiciones_x[0], alturas[0][1])]
    ruta_activa += [(posiciones_x[paso], alturas[paso][1]) for paso in range(1, 7)]
    indice_resultado = etiquetas_salida.index(etiqueta) if etiqueta in etiquetas_salida else 1
    ruta_activa.append((posiciones_x[-1], alturas[-1][indice_resultado]))
    ruta = "M " + " L ".join(f"{x} {y}" for x, y in ruta_activa)
    particulas = "".join(
        f'<circle r="4" fill="{color}" opacity="0.9">'
        f'<animateMotion dur="4s" begin="{inicio}s" repeatCount="indefinite" '
        f'path="{ruta}" /></circle>'
        for inicio in (0, 1.3, 2.6)
    ) if activo else ""

    documento = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><style>
* { box-sizing: border-box; }
body { margin: 0; color: #e9f1fb; background: transparent; font-family: system-ui, sans-serif; }
.tarjeta { padding: 17px 19px 13px; border: 1px solid #334155; border-radius: 16px;
  background: linear-gradient(135deg, #101827, #172238); overflow: hidden; }
.titulo { margin: 0 0 2px; font-size: 19px; font-weight: 700; }
.resumen { margin: 0; color: #bac8d9; font-size: 13px; }
svg { display: block; width: 100%; height: 305px; }
.enlace { stroke: #56708e; stroke-width: 1.5; stroke-opacity: .35;
  stroke-dasharray: 5 10; animation: flujo 2s linear infinite; }
.neurona { fill: #60a5fa; stroke: #bae6fd; stroke-width: 1.3;
  transform-box: fill-box; transform-origin: center;
  animation: pulso 2.3s ease-in-out infinite; }
.resultado { fill: __COLOR__; stroke: #ffffff; stroke-width: 2.5; }
.quieta .enlace, .quieta .neurona { animation: none; }
.quieta .enlace { stroke-opacity: .20; }
.quieta .neurona { fill: #52677d; stroke: #8293a8; }
.etapa { fill: #a7bad1; font-size: 13px; text-anchor: middle; }
.capa { fill: #7f9ab8; font-size: 11px; text-anchor: middle; }
.salida { fill: #a7bad1; font-size: 13px; }
.elegida { fill: __COLOR__; font-weight: 700; }
.nota { margin: 0; color: #94a3b8; font-size: 12px; }
@keyframes flujo { to { stroke-dashoffset: -30; } }
@keyframes pulso { 50% { filter: brightness(1.8); transform: scale(1.24); } }
@media (prefers-reduced-motion: reduce) { .enlace, .neurona { animation: none; } }
</style></head><body><div class="tarjeta __ESTADO__">
<p class="titulo">Así procesa la reseña la red neuronal</p>
<p class="resumen">__RESUMEN__</p>
<svg xmlns="http://www.w3.org/2000/svg" width="1020" height="305" viewBox="0 0 1020 305" role="img" aria-label="Esquema animado del texto que pasa por seis capas de DistilBERT y llega a tres clases de sentimiento">
<text class="etapa" x="85" y="65">Tokens</text>
<text class="etapa" x="480" y="65">DistilBERT · seis capas de atención</text>
<text class="etapa" x="865" y="65">Sentimiento</text>
__ENLACES__
__NEURONAS__
__PARTICULAS__
__SALIDAS__
<text class="capa" x="205" y="292">1</text><text class="capa" x="315" y="292">2</text>
<text class="capa" x="425" y="292">3</text><text class="capa" x="535" y="292">4</text>
<text class="capa" x="645" y="292">5</text><text class="capa" x="755" y="292">6</text>
</svg>
<p class="nota">Visualización didáctica del recorrido. Las luces no son activaciones reales de neuronas; los porcentajes sí provienen del modelo.</p>
</div></body></html>"""
    documento = (
        documento.replace("__COLOR__", color)
        .replace("__ESTADO__", "" if activo else "quieta")
        .replace(
            "__RESUMEN__",
            f"{tokens_usados} tokens usados · resultado: <strong>{escape(etiqueta)}</strong>"
            if activo else "Escribe una reseña para activar el recorrido.",
        )
        .replace("__ENLACES__", "".join(enlaces))
        .replace("__NEURONAS__", "".join(neuronas))
        .replace("__PARTICULAS__", particulas)
        .replace("__SALIDAS__", "".join(salidas))
    )
    components.html(documento, height=405, scrolling=False)


st.set_page_config(page_title="ACA 2 · Sentimiento de reseñas", page_icon="🛒", layout="wide")
st.title("🛒 Sentimiento de reseñas de productos")
st.write(
    "Escribe una reseña en inglés o español. DistilBERT estima si la opinión es "
    "negativa, neutra o positiva."
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

col_entrada, col_graficas = st.columns([0.4, 0.6], gap="large")

with col_entrada:
    st.subheader("Tu reseña")
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
    ejemplo = st.selectbox(
        "Elige un ejemplo o escribe tu reseña", list(ejemplos), key=f"ejemplo_{idioma}"
    )
    texto = st.text_area(
        "Texto de la reseña",
        value=ejemplos[ejemplo],
        height=190,
        placeholder=texto_guia,
        key=f"resena_{idioma}_{ejemplo}",
    )
    st.caption("Las gráficas se actualizan al salir del campo de texto o elegir un ejemplo.")

etiqueta = None
probabilidades = None
tokens_usados = 0
if texto.strip():
    texto_modelo = texto.strip()
    try:
        with col_entrada, st.spinner("Analizando la reseña..."):
            if idioma == "Español":
                tokenizador_traduccion, modelo_traduccion = cargar_traductor()
                texto_modelo, truncado = traducir_al_ingles(
                    texto_modelo, tokenizador_traduccion, modelo_traduccion
                )
            max_length = int(configuracion["max_length"])
            longitud = len(tokenizador(texto_modelo, truncation=False)["input_ids"])
            etiqueta, probabilidades = predecir(
                texto_modelo, tokenizador, modelo, max_length
            )
            tokens_usados = min(longitud, max_length)
    except Exception as exc:
        etiqueta = None
        probabilidades = None
        tokens_usados = 0
        with col_entrada:
            st.error("No se pudo analizar la reseña. Revisa los registros de la app.")
            st.exception(exc)
    else:
        with col_entrada:
            if idioma == "Español":
                if truncado:
                    st.warning("La reseña supera el límite del traductor; se usó solo el comienzo.")
                with st.expander("Ver traducción usada por el modelo"):
                    st.write(texto_modelo)
                    st.caption("La ruta en español todavía no tiene una evaluación cuantitativa propia.")
            if longitud > max_length:
                st.info(f"El clasificador usó los primeros {max_length} tokens del texto en inglés.")
            st.subheader(f"Resultado: {etiqueta}")

with col_graficas:
    mostrar_recorrido_animado(etiqueta, probabilidades, tokens_usados)
    st.subheader("Probabilidades por sentimiento")
    if probabilidades is not None:
        tabla = pd.DataFrame(
            {"Sentimiento": list(probabilidades), "Probabilidad": list(probabilidades.values())}
        )
        st.bar_chart(tabla, x="Sentimiento", y="Probabilidad", y_label="Probabilidad")
        st.caption(
            "Estas probabilidades expresan la confianza relativa del modelo; "
            "no garantizan que la clasificación sea correcta."
        )
    else:
        st.caption("Las barras aparecerán cuando escribas una reseña.")

with st.expander("Cómo interpretar este proyecto"):
    st.write(
        "En 9.098 reseñas de prueba en inglés, el modelo obtuvo 93,58 % de exactitud y "
        "F1 macro de 0,6720. Reconoce mejor las opiniones positivas; las neutras "
        "son las más difíciles (F1 = 0,3283). Las etiquetas de entrenamiento "
        "provienen de las calificaciones por estrellas y son aproximadas. "
        "Estas métricas no miden el rendimiento de la ruta de traducción desde español."
    )
