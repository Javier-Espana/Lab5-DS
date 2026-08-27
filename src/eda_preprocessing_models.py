import os
import re
import string
import shutil
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

def ensure_nltk():
    for res in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass

ensure_nltk()
stop_words = set(stopwords.words('english')).union({'u', 'im', 'amp', 'via', 'get', 'would', 'one', 'like', 'dont', 'cant', 'also'})
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return ''
    # 1. Conversion a minusculas
    text = text.lower()
    # 2. Eliminacion de URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Eliminacion de entidades HTML
    text = re.sub(r'&[a-z]+;', ' ', text)
    # 4. Eliminacion de menciones (@usuario)
    text = re.sub(r'@\w+', '', text)
    # 5. Tratamiento de hashtags (#terremoto -> terremoto)
    text = re.sub(r'#(\w+)', r'\1', text)
    # 6. Preservacion de codigo 911 y eliminacion de otros digitos
    text = re.sub(r'\b911\b', ' emergency911 ', text)
    text = re.sub(r'\d+', '', text)
    # 7. Remocion de signos de puntuacion y caracteres especiales
    text = re.sub(r'[' + re.escape(string.punctuation) + ']', ' ', text)
    # 8. Remocion de caracteres no ascii (emoticones y simbolos)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 9. Tokenizacion, filtrado de stopwords y lemmatizacion
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

def run_part1_eda():
    print("=== EJECUTANDO PARTE 1: ANALISIS EXPLORATORIO DE DATOS ===")
    os.makedirs('docs/figures', exist_ok=True)
    sns.set_theme(style='whitegrid')
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Dimensiones Train: {train_df.shape}")
    print(f"Dimensiones Test: {test_df.shape}")
    print("Valores nulos en Train:\n", train_df.isnull().sum())
    
    # 1. Distribucion del target
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
    plt.savefig('docs/figures/target_distribution.png', dpi=300)
    plt.close()
    
    # 2. Valores faltantes
    missing_data = pd.DataFrame({
        'Columna': train_df.columns,
        'Faltantes': train_df.isnull().sum().values,
        'Porcentaje': (train_df.isnull().sum().values / len(train_df)) * 100
    })
    plt.figure(figsize=(7, 4))
    sns.barplot(x='Columna', y='Porcentaje', data=missing_data, palette='Blues_r', hue='Columna', legend=False)
    plt.title('Porcentaje de Valores Faltantes por Columna (Train)')
    plt.ylabel('Porcentaje (%)')
    plt.xlabel('Columna')
    plt.tight_layout()
    plt.savefig('docs/figures/missing_values.png', dpi=300)
    plt.close()
    
    # 3. Top Keywords
    top_disaster = train_df[train_df['target'] == 1]['keyword'].value_counts().head(10)
    top_nondisaster = train_df[train_df['target'] == 0]['keyword'].value_counts().head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x=top_disaster.values, y=top_disaster.index, ax=axes[0], palette='Reds_r', hue=top_disaster.index, legend=False)
    axes[0].set_title('Top 10 Keywords en Tweets de Desastres (Target = 1)')
    axes[0].set_xlabel('Frecuencia')
    
    sns.barplot(x=top_nondisaster.values, y=top_nondisaster.index, ax=axes[1], palette='Blues_r', hue=top_nondisaster.index, legend=False)
    axes[1].set_title('Top 10 Keywords en Tweets No Desastres (Target = 0)')
    axes[1].set_xlabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('docs/figures/top_keywords.png', dpi=300)
    plt.close()
    
    # 4. Longitud de caracteres y palabras
    train_df['char_count'] = train_df['text'].astype(str).apply(len)
    train_df['word_count'] = train_df['text'].astype(str).apply(lambda x: len(x.split()))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=train_df, x='char_count', hue='target', kde=True, ax=axes[0], palette=['#2b5c8f', '#d9534f'], element='step')
    axes[0].set_title('Distribucion de Longitud de Caracteres')
    axes[0].set_xlabel('Numero de Caracteres')
    axes[0].set_ylabel('Frecuencia')
    
    sns.histplot(data=train_df, x='word_count', hue='target', kde=True, ax=axes[1], palette=['#2b5c8f', '#d9534f'], element='step')
    axes[1].set_title('Distribucion de Conteo de Palabras')
    axes[1].set_xlabel('Numero de Palabras')
    axes[1].set_ylabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('docs/figures/text_length_distribution.png', dpi=300)
    plt.close()
    
    return train_df, test_df

def get_top_ngrams(corpus, n=1, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:top_k], columns=['ngram', 'count'])

def run_part2_preprocessing_and_ngrams(train_df):
    print("=== EJECUTANDO PARTE 2: PREPROCESAMIENTO Y ANALISIS DE N-GRAMAS ===")
    train_df['cleaned_text'] = train_df['text'].apply(clean_text)
    
    # Nubes de palabras
    disaster_text = ' '.join(train_df[train_df['target'] == 1]['cleaned_text'])
    nondisaster_text = ' '.join(train_df[train_df['target'] == 0]['cleaned_text'])
    
    wc_disaster = WordCloud(width=800, height=400, background_color='white', colormap='Reds', max_words=100, random_state=42).generate(disaster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_disaster, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras - Tweets de Desastres Reales (Target = 1)', fontsize=14)
    plt.tight_layout()
    plt.savefig('docs/figures/wordcloud_disaster.png', dpi=300)
    plt.close()
    
    wc_nondisaster = WordCloud(width=800, height=400, background_color='white', colormap='Blues', max_words=100, random_state=42).generate(nondisaster_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc_nondisaster, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras - Tweets No Desastres (Target = 0)', fontsize=14)
    plt.tight_layout()
    plt.savefig('docs/figures/wordcloud_nondisaster.png', dpi=300)
    plt.close()
    
    # N-gramas
    disaster_corpus = train_df[train_df['target'] == 1]['cleaned_text']
    nondisaster_corpus = train_df[train_df['target'] == 0]['cleaned_text']
    
    # Unigramas
    top_uni_disaster = get_top_ngrams(disaster_corpus, n=1, top_k=15)
    top_uni_nondisaster = get_top_ngrams(nondisaster_corpus, n=1, top_k=15)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='count', y='ngram', data=top_uni_disaster, ax=axes[0], palette='Reds_r', hue='ngram', legend=False)
    axes[0].set_title('Top 15 Unigramas en Desastres Reales')
    axes[0].set_xlabel('Frecuencia')
    
    sns.barplot(x='count', y='ngram', data=top_uni_nondisaster, ax=axes[1], palette='Blues_r', hue='ngram', legend=False)
    axes[1].set_title('Top 15 Unigramas en No Desastres')
    axes[1].set_xlabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('docs/figures/top_unigrams.png', dpi=300)
    plt.close()
    
    # Bigramas
    top_bi_disaster = get_top_ngrams(disaster_corpus, n=2, top_k=15)
    top_bi_nondisaster = get_top_ngrams(nondisaster_corpus, n=2, top_k=15)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='count', y='ngram', data=top_bi_disaster, ax=axes[0], palette='Reds_r', hue='ngram', legend=False)
    axes[0].set_title('Top 15 Bigramas en Desastres Reales')
    axes[0].set_xlabel('Frecuencia')
    
    sns.barplot(x='count', y='ngram', data=top_bi_nondisaster, ax=axes[1], palette='Blues_r', hue='ngram', legend=False)
    axes[1].set_title('Top 15 Bigramas en No Desastres')
    axes[1].set_xlabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('docs/figures/top_bigrams.png', dpi=300)
    plt.close()
    
    print("Figuras de Parte 2 generadas con exito.")
    return train_df

if __name__ == '__main__':
    train_df, test_df = run_part1_eda()
    train_df = run_part2_preprocessing_and_ngrams(train_df)
