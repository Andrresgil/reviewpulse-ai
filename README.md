# ACA 2 · Sentimiento de reseñas

## Aplicación

La aplicación acepta reseñas en inglés y español. Las reseñas en inglés se envían
directamente al clasificador [DistilBERT entrenado para este proyecto](https://huggingface.co/Andrresgil/aca2-sentimiento-retail).
Para español, la aplicación traduce el texto al inglés con
[Helsinki-NLP/opus-mt-es-en](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)
y muestra la traducción antes de presentar la clasificación. La primera consulta
en español puede tardar mientras descarga el traductor.

## Resultado del modelo

En 9.098 reseñas de prueba **en inglés**: exactitud 93,58 %, F1 macro 0,6720.
El F1 de la clase neutra fue 0,3283. Estos resultados no miden la ruta en español,
que requiere una evaluación separada con reseñas etiquetadas en español. Los datos
y el entrenamiento se documentan en el cuaderno ACA 2.
