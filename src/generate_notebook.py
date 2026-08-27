import nbformat as nbf
import os

def create_part1_notebook():
    nb = nbf.v4.new_notebook()
    cells = []

    # Title & Metadata
    cells.append(nbf.v4.new_markdown_cell("""# Laboratorio 5: Mineria de Textos y Analisis de Sentimiento
## Entrega Parcial (Avances)
**Curso:** Data Science - CC3084  
**Universidad del Valle de Guatemala**  
**Integrantes:**
- Javier Eduardo Espana (23204)
- Angel Esteban Esquit (23249)
- Bryan Gabriel Aguilar (23354)

---
### Contenido de la Entrega Parcial:
1. Descarga y Carga de Datos (Kagglehub y verificacion de datos).
2. Descripcion y Analisis Exploratorio de Datos (EDA).
3. Limpieza y Preprocesamiento de Texto.
4. Analisis de N-gramas (Unigramas, Bigramas, Trigramas) y Frecuencias.
5. Modelos Preliminares de Clasificacion y Evaluacion."""))

    # Section 1: Descarga y Carga
    cells.append(nbf.v4.new_markdown_cell("""## 1. Descarga y Carga de Datos
En esta seccion se utiliza la libreria `kagglehub` para acceder al conjunto de datos de la competencia *Natural Language Processing with Disaster Tweets* y se cargan los archivos `train.csv` y `test.csv`."""))

    cells.append(nbf.v4.new_code_cell("""import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Descarga usando kagglehub con soporte de respaldo local
os.makedirs('../data', exist_ok=True)
try:
    import kagglehub
    path = kagglehub.competition_download('nlp-getting-started')
    print('Path to competition files:', path)
    if os.path.exists(path):
        for f in os.listdir(path):
            shutil.copy2(os.path.join(path, f), os.path.join('../data', f))
except Exception as e:
    print('kagglehub omitido o sin autenticacion interactiva, cargando desde ../data/:', e)

train_df = pd.read_csv('../data/train.csv')
test_df = pd.read_csv('../data/test.csv')

print(f'Dimensiones Train: {train_df.shape}')
print(f'Dimensiones Test: {test_df.shape}')
train_df.head()"""))

    # Section 2: EDA
    cells.append(nbf.v4.new_markdown_cell("""## 2. Descripcion y Analisis Exploratorio de Datos (EDA)
Se analizan los tipos de variables, presencia de valores nulos, distribucion del target y longitudes de texto."""))

    cells.append(nbf.v4.new_code_cell("""print("Informacion del dataset de entrenamiento:")
print(train_df.info())
print("\\nValores nulos por columna:")
print(train_df.isnull().sum())"""))

    cells.append(nbf.v4.new_code_cell("""# Distribucion de la variable objetivo
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

    cells.append(nbf.v4.new_code_cell("""# Analisis de Longitud de Caracteres y Palabras
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

    cells.append(nbf.v4.new_code_cell("""# Top Keywords por Categoria
top_disaster = train_df[train_df['target'] == 1]['keyword'].value_counts().head(10)
top_nondisaster = train_df[train_df['target'] == 0]['keyword'].value_counts().head(10)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(x=top_disaster.values, y=top_disaster.index, ax=axes[0], palette='Reds_r', hue=top_disaster.index, legend=False)
axes[0].set_title('Top 10 Keywords en Desastres (Target = 1)')

sns.barplot(x=top_nondisaster.values, y=top_nondisaster.index, ax=axes[1], palette='Blues_r', hue=top_nondisaster.index, legend=False)
axes[1].set_title('Top 10 Keywords en No Desastres (Target = 0)')
plt.tight_layout()
plt.show()"""))

    nb.cells = cells
    with open('notebooks/avance_laboratorio5.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print('Notebook Part 1 generated successfully.')

if __name__ == '__main__':
    create_part1_notebook()
