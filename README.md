# Laboratorio 5: Mineria de Textos y Analisis de Sentimiento

Universidad del Valle de Guatemala  
Facultad de Ingenieria  
Departamento de Ciencias de la Computacion  
CC3084 - Data Science  
Semestre II - 2026  

## Integrantes del Grupo
- Javier Eduardo Espana (23204)
- Angel Esteban Esquit (23249)
- Roberto José Barreda (23354)

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
|   |   |-- confusion_matrices_preliminary.png
|   |   |-- missing_values.png
|   |   |-- models_comparison_metrics.png
|   |   |-- preliminary_metrics.csv
|   |   |-- roc_curves_preliminary.png
|   |   |-- target_distribution.png
|   |   |-- text_length_distribution.png
|   |   |-- top_bigrams.png
|   |   |-- top_keywords.png
|   |   |-- top_unigrams.png
|   |   |-- wordcloud_disaster.png
|   |   |-- wordcloud_nondisaster.png
|   |-- Lab5Informe.tex
|   |-- Lab5Informe.pdf
|   |-- Laboratorio 5. Mineria de Textos y analisis de sentimiento. 2026.pdf
|-- notebooks/
|   |-- avance_laboratorio5.ipynb
|-- src/
|   |-- download_data.py
|   |-- eda_preprocessing_models.py
|   |-- generate_notebook.py
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

## Reproducibilidad de los Avances

Para ejecutar todo el pipeline de analisis exploratorio, preprocesamiento y modelado preliminar:

```bash
# Ejecutar pipeline completo y generar graficas
python3 src/eda_preprocessing_models.py

# Construir y ejecutar el notebook interactivo con todas las salidas
python3 src/generate_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/avance_laboratorio5.ipynb

# Compilar el informe en LaTeX
cd docs && pdflatex Lab5Informe.tex && cd ..
```

---

## Resumen de Avances y Contribuciones

La entrega parcial de avances se estructuro en tres etapas:

1. **Parte 1 (Javier-Espana):**
   - Configuracion inicial del repositorio, `.gitignore` y `requirements.txt`.
   - Descarga e integracion de datos en `data/` con soporte para `kagglehub`.
   - Analisis Exploratorio de Datos (EDA): revision de dimensiones (7,613 entrenamiento, 3,263 prueba), analisis de valores nulos (33.3% en `location`, 0.8% en `keyword`), balance de clases (57% no desastre vs 43% desastre real) y distribucion de longitud de texto.
   - Creacion de la estructura base del informe en LaTeX.

2. **Parte 2 (AngelEsquit):**
   - Diseno e implementacion del pipeline de limpieza y preprocesamiento de texto (minusculas, remocion de URLs, entidades HTML, menciones `@`, hashtags preservando token, eliminacion de puntuacion, filtrado de stopwords y lematizacion con WordNet).
   - Tratamiento de digitos numericos preservando selectivamente el token de emergencia `911` (`emergency911`).
   - Analisis de n-gramas (unigramas, bigramas, trigramas), generacion de nubes de palabras (`WordCloud`) e histogramas comparativos.
   - Discusion sobre terminos ambiguos compartidos (`fire`, `body`) y la necesidad del contexto.

3. **Parte 3 (bar23354):**
   - Vectorizacion TF-IDF con unigramas y bigramas (`ngram_range=(1,2)`), escalamiento sublineal (`sublinear_tf=True`) y vocabulario de 10,000 caracteristicas.
   - Entrenamiento y evaluacion comparativa de modelos preliminares: `Multinomial Naive Bayes`, `Complement Naive Bayes`, `Logistic Regression` y `Linear SVM (SGD)`.
   - Generacion de matrices de confusion, curvas ROC-AUC y tablas comparativas de metricas.
   - Ejecucion completa del Jupyter Notebook y compilacion del informe final de avances en PDF (`docs/Lab5Informe.pdf`).

