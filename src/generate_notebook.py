import nbformat as nbf
import os

def build_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Titulo y Metadatos
    cells.append(nbf.v4.new_markdown_cell("""# Laboratorio 5: Mineria de Textos y Analisis de Sentimiento
## Entrega Parcial (Avances)
**Curso:** Data Science - CC3084  
**Universidad del Valle de Guatemala**  
**Integrantes:**
- Javier Eduardo Espana (23204)
- Angel Esteban Esquit (23249)
- Bryan Gabriel Aguilar (23354)

---
### Estructura de la Entrega Parcial:
1. **Descarga y Carga de Datos:** Descarga mediante `kagglehub` e integracion de respaldo.
2. **Analisis Exploratorio de Datos (EDA):** Evaluacion de dimensiones, tipos, valores nulos, distribucion de target y longitud de texto.
3. **Limpieza y Preprocesamiento de Texto:** Pipeline de normalizacion, remocion de ruido, filtrado de stopwords y tratamiento especifico de tokens numericos (911).
4. **Analisis de N-gramas y Frecuencias:** Frecuencias de unigramas, bigramas y trigramas, nubes de palabras, histogramas y analisis de solapamiento lexico.
5. **Modelos Preliminares de Clasificacion:** Vectorizacion con TF-IDF, entrenamiento y evaluacion de modelos baseline (Naive Bayes, Regresion Logistica, Linear SVM)."""))

    # 1. Descarga y Carga
    cells.append(nbf.v4.new_markdown_cell("""## 1. Descarga y Carga de Datos
Se realiza la integracion con la libreria `kagglehub` y se cargan los conjuntos de datos `train.csv` y `test.csv`."""))

    cells.append(nbf.v4.new_code_cell("""import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Descarga usando kagglehub con respaldo local
os.makedirs('../data', exist_ok=True)
try:
    import kagglehub
    path = kagglehub.competition_download('nlp-getting-started')
    print('Ruta de competencia kagglehub:', path)
    if os.path.exists(path):
        for f in os.listdir(path):
            shutil.copy2(os.path.join(path, f), os.path.join('../data', f))
except Exception as e:
    print('kagglehub omitido o no autenticado, cargando desde ../data/:', e)

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print(f'Dimensiones Train: {train_df.shape}')
print(f'Dimensiones Test: {test_df.shape}')
train_df.head()"""))

    # 2. EDA
    cells.append(nbf.v4.new_markdown_cell("""## 2. Descripcion y Analisis Exploratorio de Datos (EDA)
Se inspeccionan las variables, distribucion de clases, valores faltantes y diferencias en la longitud de caracteres y palabras."""))

    cells.append(nbf.v4.new_code_cell("""print("Informacion del dataset de entrenamiento:")
print(train_df.info())
print("\\nValores nulos por columna en Train:")
print(train_df.isnull().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# 2.1 Distribucion de la variable objetivo
plt.figure(figsize=(6, 4))
counts = train_df['target'].value_counts()
ax = sns.barplot(x=counts.index, y=counts.values, palette=['#2b5c8f', '#d9534f'], hue=counts.index, legend=False)
plt.title('Distribucion de la Variable Objetivo (Target)')
plt.xlabel('Clase (0 = No Desastre, 1 = Desastre Real)')
plt.ylabel('Cantidad de Tweets')
plt.xticks([0, 1], ['No Desastre (0)', 'Desastre Real (1)'])
for p in ax.patches:
    h = p.get_height()
    ax.annotate(f'{int(h)} ({h/len(train_df)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom', fontsize=10, xytext=(0, 3), textcoords='offset points')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 2.2 Distribucion de Caracteres y Conteo de Palabras
train_df['char_count'] = train_df['text'].astype(str).apply(len)
train_df['word_count'] = train_df['text'].astype(str).apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=train_df, x='char_count', hue='target', kde=True, ax=axes[0], palette=['#2b5c8f', '#d9534f'], element='step')
axes[0].set_title('Distribucion de Longitud de Caracteres')
axes[0].set_xlabel('Numero de Caracteres')

sns.histplot(data=train_df, x='word_count', hue='target', kde=True, ax=axes[1], palette=['#2b5c8f', '#d9534f'], element='step')
axes[1].set_title('Distribucion de Conteo de Palabras')
axes[1].set_xlabel('Numero de Palabras')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 2.3 Top Keywords
top_disaster = train_df[train_df['target'] == 1]['keyword'].value_counts().head(10)
top_nondisaster = train_df[train_df['target'] == 0]['keyword'].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=top_disaster.values, y=top_disaster.index, ax=axes[0], palette='Reds_r', hue=top_disaster.index, legend=False)
axes[0].set_title('Top 10 Keywords en Desastres (Target = 1)')

sns.barplot(x=top_nondisaster.values, y=top_nondisaster.index, ax=axes[1], palette='Blues_r', hue=top_nondisaster.index, legend=False)
axes[1].set_title('Top 10 Keywords en No Desastres (Target = 0)')
plt.tight_layout()
plt.show()"""))

    # 3. Preprocesamiento
    cells.append(nbf.v4.new_markdown_cell("""## 3. Limpieza y Preprocesamiento de Datos (Ejercicio 3)
Se desarrolla un pipeline de preprocesamiento modular utilizando las bibliotecas `re`, `string` y `nltk`:
1. **Conversion a minusculas:** Estandarizacion del espacio de texto.
2. **Eliminacion de URLs y entidades HTML:** Remocion de enlaces y secuencias como `&amp;`, `&lt;`, `&gt;`.
3. **Manejo de menciones (@) y hashtags (#):** Se eliminan las menciones de usuario ya que no aportan semantica de desastre, y se preserva el texto interior de los hashtags (`#earthquake` -> `earthquake`).
4. **Tratamiento de caracteres especiales y emoticones:** Remocion de puntuacion y caracteres no ASCII.
5. **Filtrado de stopwords:** Eliminacion de articulos, preposiciones, conjunciones y ruido especifico de redes sociales (`amp`, `via`, etc.).
6. **Tratamiento de numeros y codigo 911:** Los digitos arbitrarios se descartan, pero el token clave `911` se conserva mapeado a `emergency911` dado su alto valor diagnostico.
7. **Lemmatizacion:** Normalizacion morfologica con `WordNetLemmatizer`."""))

    cells.append(nbf.v4.new_code_cell("""for res in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
    nltk.download(res, quiet=True)

stop_words = set(stopwords.words('english')).union({'u', 'im', 'amp', 'via', 'get', 'would', 'one', 'like', 'dont', 'cant', 'also'})
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # 1. Minusculas
    text = text.lower()
    # 2. URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Entidades HTML
    text = re.sub(r'&[a-z]+;', ' ', text)
    # 4. Menciones @usuario
    text = re.sub(r'@\w+', '', text)
    # 5. Hashtags (#tag -> tag)
    text = re.sub(r'#(\w+)', r'\1', text)
    # 6. Preservar 911 y remover otros digitos
    text = re.sub(r'\\b911\\b', ' emergency911 ', text)
    text = re.sub(r'\\d+', '', text)
    # 7. Puntuacion
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    # 8. Remover no-ascii
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 9. Tokenizar, remover stopwords y lemmatizar
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

train_df['cleaned_text'] = train_df['text'].apply(clean_text)

# Comparacion antes y despues del preprocesamiento
comparison_sample = train_df[['text', 'cleaned_text', 'target']].head(10)
comparison_sample"""))

    # 4. N-gramas y Frecuencias
    cells.append(nbf.v4.new_markdown_cell("""## 4. Frecuencia de Palabras y Analisis de N-gramas (Ejercicios 4 y 5)
Se analizan unigramas, bigramas y trigramas para ambas categorias, evaluando la capacidad predictiva y la necesidad del contexto."""))

    cells.append(nbf.v4.new_code_cell("""# 4.1 Nubes de Palabras (WordCloud)
disaster_text = ' '.join(train_df[train_df['target'] == 1]['cleaned_text'])
nondisaster_text = ' '.join(train_df[train_df['target'] == 0]['cleaned_text'])

wc_disaster = WordCloud(width=800, height=400, background_color='white', colormap='Reds', max_words=100, random_state=42).generate(disaster_text)
wc_nondisaster = WordCloud(width=800, height=400, background_color='white', colormap='Blues', max_words=100, random_state=42).generate(nondisaster_text)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(wc_disaster, interpolation='bilinear')
axes[0].axis('off')
axes[0].set_title('Nube de Palabras - Desastres Reales (Target = 1)', fontsize=14)

axes[1].imshow(wc_nondisaster, interpolation='bilinear')
axes[1].axis('off')
axes[1].set_title('Nube de Palabras - No Desastres (Target = 0)', fontsize=14)
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_code_cell("""# 4.2 Analisis de Unigramas y Bigramas mas Frecuentes
def get_top_ngrams(corpus, n=1, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:top_k], columns=['ngram', 'count'])

disaster_corpus = train_df[train_df['target'] == 1]['cleaned_text']
nondisaster_corpus = train_df[train_df['target'] == 0]['cleaned_text']

top_uni_disaster = get_top_ngrams(disaster_corpus, n=1, top_k=15)
top_uni_nondisaster = get_top_ngrams(nondisaster_corpus, n=1, top_k=15)

top_bi_disaster = get_top_ngrams(disaster_corpus, n=2, top_k=15)
top_bi_nondisaster = get_top_ngrams(nondisaster_corpus, n=2, top_k=15)

# Visualizacion de Unigramas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='count', y='ngram', data=top_uni_disaster, ax=axes[0], palette='Reds_r', hue='ngram', legend=False)
axes[0].set_title('Top 15 Unigramas en Desastres Reales')
axes[0].set_xlabel('Frecuencia')

sns.barplot(x='count', y='ngram', data=top_uni_nondisaster, ax=axes[1], palette='Blues_r', hue='ngram', legend=False)
axes[1].set_title('Top 15 Unigramas en No Desastres')
axes[1].set_xlabel('Frecuencia')
plt.tight_layout()
plt.show()

# Visualizacion de Bigramas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(x='count', y='ngram', data=top_bi_disaster, ax=axes[0], palette='Reds_r', hue='ngram', legend=False)
axes[0].set_title('Top 15 Bigramas en Desastres Reales')
axes[0].set_xlabel('Frecuencia')

sns.barplot(x='count', y='ngram', data=top_bi_nondisaster, ax=axes[1], palette='Blues_r', hue='ngram', legend=False)
axes[1].set_title('Top 15 Bigramas en No Desastres')
axes[1].set_xlabel('Frecuencia')
plt.tight_layout()
plt.show()"""))

    cells.append(nbf.v4.new_markdown_cell("""### Discusion y Hallazgos sobre el Analisis de N-gramas:
1. **Palabras altamente predictivas de desastre:** Terminos como `fire`, `california`, `suicide`, `disaster`, `police`, `kill`, `storm` y `flood` tienen una frecuencia notablemente superior en la clase positiva.
2. **Palabras compartidas y ambiguedad:** Palabras como `fire`, `body`, `building` y `people` se presentan con alta frecuencia en ambas categorias. Por ejemplo, `fire` se utiliza de forma literal en desastres (*"wildfire burning homes"*) y de forma figurativa en no desastres (*"this song is fire"*). De igual forma, `body` aparece en desastres asociado a fatalidades y en no desastres asociado a moda o vestimenta (*"cross body bag"*).
3. **Importancia del Contexto (Bigramas y Trigramas):** Los bigramas como `suicide bomber`, `california wildfire`, `severe thunderstorm` y `oil spill` eliminan la ambiguedad y proporcionan senales contextuales contundentes para los clasificadores."""))

    nb.cells = cells
    with open('notebooks/avance_laboratorio5.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('Notebook Part 1 & 2 generated successfully.')

if __name__ == '__main__':
    build_notebook()
