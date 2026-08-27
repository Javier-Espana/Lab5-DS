# Laboratorio 5: Mineria de Textos y Analisis de Sentimiento

Universidad del Valle de Guatemala  
Facultad de Ingenieria  
Departamento de Ciencias de la Computacion  
CC3084 - Data Science  
Semestre II - 2026  

## Integrantes del Grupo
- Javier Eduardo Espana (23204)
- Angel Esteban Esquit (23249)
- Bryan Gabriel Aguilar (23354)

---

## Descripcion del Proyecto

El presente proyecto aborda el procesamiento de lenguaje natural (NLP) y mineria de textos aplicado a la clasificacion de tweets del conjunto de datos **Natural Language Processing with Disaster Tweets** de Kaggle. El objetivo primordial consiste en clasificar automaticamente si un determinado tweet hace referencia a un desastre real (`target = 1`) o si corresponde a un mensaje de otra indole (`target = 0`).

---

## Estructura del Repositorio

```text
Lab5-DS/
|-- data/
|   |-- train.csv
|   |-- test.csv
|-- docs/
|   |-- figures/
|   |-- Lab5Informe.tex
|   |-- Lab5Informe.pdf
|   |-- Laboratorio 5. Mineria de Textos y analisis de sentimiento. 2026.pdf
|-- notebooks/
|   |-- avance_laboratorio5.ipynb
|-- src/
|   |-- download_data.py
|   |-- eda_preprocessing_models.py
|-- requirements.txt
|-- .gitignore
|-- README.md
```

---

## Requisitos e Instalacion

Para instalar las dependencias necesarias en el entorno de Python:

```bash
pip install -r requirements.txt
```

---

## Descarga del Conjunto de Datos

El dataset puede descargarse directamente utilizando la libreria `kagglehub` o mediante el script automatizado:

```python
import kagglehub

# Descarga de la ultima version
path = kagglehub.competition_download('nlp-getting-started')
print("Ruta de archivos de la competencia:", path)
```

O ejecutando:
```bash
python3 src/download_data.py
```

---

## Avances - Entrega Parcial

La entrega parcial abarca las siguientes fases fundamentales:
1. **Descripcion de los datos y analisis exploratorio:** Revision de dimensiones, valores nulos, distribucion de clases, longitud de caracteres, conteo de palabras y analisis de atributos `keyword` y `location`.
2. **Preprocesamiento y limpieza de texto:** Pipeline exhaustivo de normalizacion (minusculas, remocion de URLs, menciones, hashtags, signos de puntuacion, filtrado de stopwords y tratamiento especifico de tokens numericos como 911).
3. **Analisis de N-gramas:** Generacion de unigramas, bigramas y trigramas, nubes de palabras, histogramas comparativos y discusion de terminos ambiguos/compartidos.
4. **Modelos preliminares de clasificacion:** Vectorizacion con TF-IDF, entrenamiento y evaluacion de modelos baseline (Naive Bayes, Regresion Logistica, Linear SVM).
