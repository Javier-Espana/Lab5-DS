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
        except Exception as e:
            pass

def run_part1_eda():
    print("--- EJECUTANDO PARTE 1: ANALISIS EXPLORATORIO DE DATOS ---")
    os.makedirs('docs/figures', exist_ok=True)
    sns.set_theme(style='whitegrid')
    
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    print(f"Dimensiones Train: {train_df.shape}")
    print(f"Dimensiones Test: {test_df.shape}")
    print("Valores nulos en Train:")
    print(train_df.isnull().sum())
    
    # 1. Distribucion de la variable objetivo
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
    
    print("Figuras de Parte 1 generadas con exito.")
    return train_df, test_df

if __name__ == '__main__':
    ensure_nltk()
    run_part1_eda()
