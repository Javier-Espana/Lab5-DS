# Laboratorio 5: Mineria de Textos y Analisis de Sentimiento

CC3084 Data Science, Universidad del Valle de Guatemala. Semestre II 2026.

Clasificacion de tweets del dataset [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started) segun se refieran a un desastre real (`target = 1`) o no (`target = 0`).

## Integrantes

- Javier Eduardo España (23361)
- Angel Esteban Esquit (23221)
- Roberto Jose Barreda (23354)

## Instalacion

```bash
pip install -r requirements.txt
```

## Uso

```bash
# Descarga del dataset (opcional, ya incluido en data/)
python src/download_data.py

# Pipeline completo: EDA, preprocesamiento, n-gramas, modelos,
# funcion de clasificacion, sentimiento y variable de negatividad
python src/eda_preprocessing_models.py

# Notebook con todas las salidas
jupyter nbconvert --to notebook --execute --inplace notebooks/Laboratorio5.ipynb
```

El script guarda las figuras y tablas en `docs/figures/` y el modelo final en `models/`.

## Informe

`docs/Lab5Informe.pdf`
