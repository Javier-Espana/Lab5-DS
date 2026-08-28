import os
import re
import string
import shutil
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import nltk
from scipy.sparse import hstack, csr_matrix
from scipy.stats import mannwhitneyu
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
    # 5. Tratamiento de hashtags (#tag -> tag)
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
    df = pd.DataFrame(words_freq[:top_k], columns=['ngram', 'count'])
    total = int(sum_words.sum())
    df['probability'] = df['count'] / total
    return df

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

    # Trigramas
    top_tri_disaster = get_top_ngrams(disaster_corpus, n=3, top_k=15)
    top_tri_nondisaster = get_top_ngrams(nondisaster_corpus, n=3, top_k=15)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(x='count', y='ngram', data=top_tri_disaster, ax=axes[0], palette='Reds_r', hue='ngram', legend=False)
    axes[0].set_title('Top 15 Trigramas en Desastres Reales')
    axes[0].set_xlabel('Frecuencia')

    sns.barplot(x='count', y='ngram', data=top_tri_nondisaster, ax=axes[1], palette='Blues_r', hue='ngram', legend=False)
    axes[1].set_title('Top 15 Trigramas en No Desastres')
    axes[1].set_xlabel('Frecuencia')
    plt.tight_layout()
    plt.savefig('docs/figures/top_trigrams.png', dpi=300)
    plt.close()

    # Tabla de frecuencias y probabilidades por n-grama y clase
    tablas = []
    for n, nombre in [(1, 'unigrama'), (2, 'bigrama'), (3, 'trigrama')]:
        for corpus, clase in [(disaster_corpus, 'Desastre real'), (nondisaster_corpus, 'No desastre')]:
            t = get_top_ngrams(corpus, n=n, top_k=15).copy()
            t.insert(0, 'tipo', nombre)
            t.insert(1, 'clase', clase)
            tablas.append(t)
    ngram_table = pd.concat(tablas, ignore_index=True)
    ngram_table.to_csv('docs/figures/ngram_frequencies.csv', index=False)
    print("Frecuencias y probabilidades de n-gramas guardadas en docs/figures/ngram_frequencies.csv")

    return train_df

def run_part3_models(train_df):
    print("=== EJECUTANDO PARTE 3: MODELOS PRELIMINARES DE CLASIFICACION ===")
    
    # Division estratificada
    X = train_df['cleaned_text']
    y = train_df['target']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorizacion TF-IDF
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)
    
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
        'Complement Naive Bayes': ComplementNB(alpha=1.0),
        'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'Linear SVM (SGD)': SGDClassifier(loss='log_loss', penalty='l2', random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_val_vec)
        y_proba = model.predict_proba(X_val_vec)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_val, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
        roc = roc_auc_score(y_val, y_proba) if y_proba is not None else 0.0
        
        results.append({
            'Modelo': name,
            'Accuracy': acc,
            'Precision': p,
            'Recall': r,
            'F1-Score': f1,
            'ROC-AUC': roc
        })
        trained_models[name] = (model, y_pred, y_proba)
        
    res_df = pd.DataFrame(results)
    print("\nResultados Comparativos de Modelos:")
    print(res_df.to_string(index=False))
    
    # Guardar metricas en CSV
    res_df.to_csv('docs/figures/preliminary_metrics.csv', index=False)
    
    # 1. Barplot comparativo de metricas
    res_melted = pd.melt(res_df, id_vars=['Modelo'], var_name='Metrica', value_name='Valor')
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Modelo', y='Valor', hue='Metrica', data=res_melted, palette='tab10')
    plt.title('Comparacion de Metricas de Rendimiento - Modelos Preliminares')
    plt.ylim(0.5, 1.0)
    plt.ylabel('Puntaje')
    plt.xlabel('Modelo')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('docs/figures/models_comparison_metrics.png', dpi=300)
    plt.close()
    
    # 2. Matrices de confusion
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    for idx, (name, (model, y_pred, y_proba)) in enumerate(trained_models.items()):
        cm = confusion_matrix(y_val, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
                    xticklabels=['No Desastre', 'Desastre'], yticklabels=['No Desastre', 'Desastre'])
        axes[idx].set_title(name)
        axes[idx].set_xlabel('Predicho')
        axes[idx].set_ylabel('Real')
    plt.tight_layout()
    plt.savefig('docs/figures/confusion_matrices_preliminary.png', dpi=300)
    plt.close()
    
    # 3. Curvas ROC
    plt.figure(figsize=(8, 6))
    for name, (model, y_pred, y_proba) in trained_models.items():
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_val, y_proba)
            roc = roc_auc_score(y_val, y_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Clasificador Aleatorio (AUC = 0.5000)')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curvas ROC - Modelos Preliminares de Clasificacion')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('docs/figures/roc_curves_preliminary.png', dpi=300)
    plt.close()
    
    print("Figuras de Parte 3 generadas con exito.")
    return res_df


def run_part4_classification_function(train_df):
    """Ejercicio 7: entrena la Regresion Logistica final sobre el 100% de train."""
    print("=== EJECUTANDO PARTE 4: FUNCION DE CLASIFICACION DE TWEETS ===")

    os.makedirs('models', exist_ok=True)

    X_full = train_df['cleaned_text']
    y_full = train_df['target']

    tfidf_final = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_full_vec = tfidf_final.fit_transform(X_full)

    best_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    best_model.fit(X_full_vec, y_full)

    joblib.dump(best_model, 'models/best_model_logreg.joblib')
    joblib.dump(tfidf_final, 'models/tfidf_vectorizer.joblib')

    print("Modelo final (Regresion Logistica) y vectorizador TF-IDF guardados en 'models/'.")

    return best_model, tfidf_final


def classify_tweet(text, model=None, vectorizer=None):
    """Clasifica un tweet crudo como desastre real (1) o no (0).

    Aplica clean_text internamente. Si no se pasan model y vectorizer, los carga
    desde models/. Retorna text, cleaned_text, label, label_desc y probability.
    """
    if model is None or vectorizer is None:
        model = joblib.load('models/best_model_logreg.joblib')
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred = int(model.predict(vec)[0])
    proba = float(model.predict_proba(vec)[0, 1]) if hasattr(model, 'predict_proba') else None

    return {
        'text': text,
        'cleaned_text': cleaned,
        'label': pred,
        'label_desc': 'Desastre real' if pred == 1 else 'No es un desastre',
        'probability': proba
    }


def run_part4_demo(model=None, vectorizer=None):
    print("\n=== DEMO: CLASIFICACION DE TWEETS NUEVOS (SIN PREPROCESAR) ===")
    ejemplos = [
        "Forest fire near La Ronge Sask. Canada",
        "This new album is fire, I can't stop listening to it!",
        "BREAKING: Massive earthquake hits downtown, buildings collapsing #emergency",
        "I'm drowning in homework this week lol",
        "Call 911 immediately, there's been an explosion at the factory",
        "Just had the best burger of my life at this new restaurant downtown",
    ]
    for tweet in ejemplos:
        resultado = classify_tweet(tweet, model=model, vectorizer=vectorizer)
        print(f"\nTweet: {resultado['text']}")
        print(f"  -> Prediccion: {resultado['label_desc']} (prob. desastre = {resultado['probability']:.4f})")


# Exige que el emoticon inicie tras un espacio y cierre en limite de palabra,
# para no confundir el ':/' de las URLs con una carita.
EMOTICON_RE = re.compile(
    r"""(?:(?<=\s)|^)[:;=][\-o\*']?[\)\(\[\]DpP/\\\|\{\}@3]+(?=\s|$|[.,!?"'])"""
    r"""|(?:(?<=\s)|^)[xX]D(?=\s|$)"""
    r"""|<3"""
    r"""|[\U0001F300-\U0001FAFF☀-➿]"""
)

_sia = None
_pos_lex = None
_neg_lex = None


def ensure_sentiment_resources():
    """Carga VADER y el lexico de opinion de Hu y Liu."""
    global _sia, _pos_lex, _neg_lex
    if _sia is None:
        for res in ['vader_lexicon', 'opinion_lexicon']:
            try:
                nltk.download(res, quiet=True)
            except Exception:
                pass
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        from nltk.corpus import opinion_lexicon
        _sia = SentimentIntensityAnalyzer()
        _pos_lex = set(opinion_lexicon.positive())
        _neg_lex = set(opinion_lexicon.negative())
    return _sia, _pos_lex, _neg_lex


def count_sentiment_words(text):
    """Cuenta palabras positivas y negativas segun el lexico de opinion."""
    _, pos_lex, neg_lex = ensure_sentiment_resources()
    tokens = str(text).split()
    n_pos = sum(1 for t in tokens if t in pos_lex)
    n_neg = sum(1 for t in tokens if t in neg_lex)
    return n_pos, n_neg


def label_from_counts(n_pos, n_neg):
    if n_pos > n_neg:
        return 'positivo'
    if n_neg > n_pos:
        return 'negativo'
    return 'neutro'


def score_sentiment(raw_text, cleaned_text):
    """Conteos de palabras de opinion y puntajes VADER de un tweet."""
    sia, _, _ = ensure_sentiment_resources()
    n_pos, n_neg = count_sentiment_words(cleaned_text)
    total = n_pos + n_neg
    vs = sia.polarity_scores(str(raw_text))
    return {
        'n_pos': n_pos,
        'n_neg': n_neg,
        'lex_score': (n_pos - n_neg) / total if total else 0.0,
        'sentiment_lex': label_from_counts(n_pos, n_neg),
        'vader_neg': vs['neg'],
        'vader_neu': vs['neu'],
        'vader_pos': vs['pos'],
        'vader_compound': vs['compound'],
        'negatividad': vs['neg']
    }


def run_part5_sentiment(train_df):
    """Ejercicio 8: clasificacion de cada tweet en positivo, negativo o neutro."""
    print("=== EJECUTANDO PARTE 5: ANALISIS DE SENTIMIENTO (EJERCICIO 8) ===")
    ensure_sentiment_resources()

    scores = [score_sentiment(r, c) for r, c in zip(train_df['text'], train_df['cleaned_text'])]
    train_df = pd.concat([train_df.reset_index(drop=True), pd.DataFrame(scores)], axis=1)

    train_df['sentimiento'] = pd.cut(
        train_df['vader_compound'],
        bins=[-1.001, -0.05, 0.05, 1.0],
        labels=['negativo', 'neutro', 'positivo']
    ).astype(str)

    print("\nDistribucion por conteo de palabras (lexico de opinion):")
    print(train_df['sentiment_lex'].value_counts())
    print("\nDistribucion por VADER:")
    print(train_df['sentimiento'].value_counts())

    # Impacto de conservar los emoticones
    sia, _, _ = ensure_sentiment_resources()
    tiene_emoticon = train_df['text'].astype(str).str.contains(EMOTICON_RE)
    sin_emoticon_txt = train_df['text'].astype(str).apply(lambda t: EMOTICON_RE.sub(' ', t))
    compound_sin = sin_emoticon_txt.apply(lambda t: sia.polarity_scores(t)['compound'])
    diff = (train_df['vader_compound'] - compound_sin).abs()
    cambia = diff > 1e-9

    emoticon_stats = pd.DataFrame([{
        'tweets_totales': len(train_df),
        'tweets_con_emoticon': int(tiene_emoticon.sum()),
        'porcentaje_con_emoticon': round(tiene_emoticon.mean() * 100, 2),
        'tweets_con_puntaje_alterado': int(cambia.sum()),
        'porcentaje_puntaje_alterado': round(cambia.mean() * 100, 2),
        'diferencia_media_abs_compound': round(diff.mean(), 4),
        'diferencia_media_abs_en_afectados': round(diff[cambia].mean(), 4) if cambia.any() else 0.0
    }])
    emoticon_stats.to_csv('docs/figures/emoticon_impact.csv', index=False)
    print("\nImpacto de conservar emoticones:")
    print(emoticon_stats.to_string(index=False))

    ct = pd.crosstab(train_df['target'], train_df['sentimiento'], normalize='index') * 100
    ct = ct.reindex(columns=['negativo', 'neutro', 'positivo'])
    ct.to_csv('docs/figures/sentiment_by_target.csv')

    plt.figure(figsize=(8, 5))
    ct.plot(kind='bar', color=['#d9534f', '#9e9e9e', '#2b8f5c'], ax=plt.gca())
    plt.title('Distribucion de Sentimiento por Categoria de Tweet')
    plt.xlabel('Clase (0 = No Desastre, 1 = Desastre Real)')
    plt.ylabel('Porcentaje dentro de la clase (%)')
    plt.xticks([0, 1], ['No Desastre (0)', 'Desastre Real (1)'], rotation=0)
    plt.legend(title='Sentimiento')
    plt.tight_layout()
    plt.savefig('docs/figures/sentiment_distribution.png', dpi=300)
    plt.close()

    return train_df


def _printable(df, col='text'):
    """Sustituye caracteres no ASCII para poder imprimir en consolas Windows."""
    out = df.copy()
    out[col] = out[col].astype(str).str.encode('ascii', 'ignore').str.decode('ascii')
    return out


def run_part6_sentiment_questions(train_df):
    """Ejercicio 9: tweets mas negativos y mas positivos, negatividad por categoria."""
    print("=== EJECUTANDO PARTE 6: PREGUNTAS DE SENTIMIENTO (EJERCICIO 9) ===")
    cols = ['id', 'text', 'target', 'vader_compound', 'n_pos', 'n_neg']

    top_neg = train_df.nsmallest(10, 'vader_compound')[cols].copy()
    top_pos = train_df.nlargest(10, 'vader_compound')[cols].copy()
    for t in (top_neg, top_pos):
        t['categoria'] = t['target'].map({0: 'No desastre', 1: 'Desastre real'})

    top_neg.to_csv('docs/figures/top10_negativos.csv', index=False, encoding='utf-8')
    top_pos.to_csv('docs/figures/top10_positivos.csv', index=False, encoding='utf-8')

    print("\n9.1 Diez tweets mas negativos:")
    print(_printable(top_neg)[['text', 'categoria', 'vader_compound']].to_string(index=False, max_colwidth=70))
    print("\n9.2 Diez tweets mas positivos:")
    print(_printable(top_pos)[['text', 'categoria', 'vader_compound']].to_string(index=False, max_colwidth=70))

    neg_des = train_df.loc[train_df['target'] == 1, 'negatividad']
    neg_no = train_df.loc[train_df['target'] == 0, 'negatividad']
    u_stat, p_val = mannwhitneyu(neg_des, neg_no, alternative='greater')

    comparacion = pd.DataFrame([
        {'categoria': 'Desastre real (1)', 'n': len(neg_des),
         'negatividad_media': neg_des.mean(), 'negatividad_mediana': neg_des.median(),
         'compound_medio': train_df.loc[train_df['target'] == 1, 'vader_compound'].mean()},
        {'categoria': 'No desastre (0)', 'n': len(neg_no),
         'negatividad_media': neg_no.mean(), 'negatividad_mediana': neg_no.median(),
         'compound_medio': train_df.loc[train_df['target'] == 0, 'vader_compound'].mean()}
    ])
    comparacion['U_mannwhitney'] = u_stat
    comparacion['p_value'] = p_val
    comparacion.to_csv('docs/figures/negativity_by_target.csv', index=False)

    print("\n9.3 Negatividad por categoria:")
    print(comparacion.to_string(index=False))
    print("Mann-Whitney (desastre > no desastre): U = %.1f, p = %.3e" % (u_stat, p_val))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=train_df, x='target', y='negatividad', ax=axes[0],
                palette=['#2b5c8f', '#d9534f'], hue='target', legend=False)
    axes[0].set_title('Negatividad por Categoria')
    axes[0].set_xlabel('Clase (0 = No Desastre, 1 = Desastre Real)')
    axes[0].set_ylabel('Negatividad (VADER)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['No Desastre', 'Desastre Real'])

    sns.kdeplot(data=train_df, x='vader_compound', hue='target', fill=True, ax=axes[1],
                palette=['#2b5c8f', '#d9534f'], common_norm=False)
    axes[1].set_title('Distribucion del Puntaje Compound por Categoria')
    axes[1].set_xlabel('Puntaje compound (VADER)')
    axes[1].set_ylabel('Densidad')
    plt.tight_layout()
    plt.savefig('docs/figures/negativity_by_target.png', dpi=300)
    plt.close()

    return top_neg, top_pos, comparacion


def run_part7_negativity_model(train_df):
    """Ejercicio 10: reentrenamiento de los modelos incluyendo la negatividad."""
    print("=== EJECUTANDO PARTE 7: MODELOS CON VARIABLE DE NEGATIVIDAD (EJERCICIO 10) ===")

    X = train_df['cleaned_text']
    y = train_df['target']
    neg = train_df[['negatividad']].values

    X_train, X_val, y_train, y_val, neg_train, neg_val = train_test_split(
        X, y, neg, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec = tfidf.transform(X_val)

    X_train_neg = hstack([X_train_vec, csr_matrix(neg_train)]).tocsr()
    X_val_neg = hstack([X_val_vec, csr_matrix(neg_val)]).tocsr()

    def build_models():
        return {
            'Multinomial Naive Bayes': MultinomialNB(alpha=1.0),
            'Complement Naive Bayes': ComplementNB(alpha=1.0),
            'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            'Linear SVM (SGD)': SGDClassifier(loss='log_loss', penalty='l2', random_state=42)
        }

    filas = []
    for etiqueta, Xtr, Xva in [('Sin negatividad', X_train_vec, X_val_vec),
                               ('Con negatividad', X_train_neg, X_val_neg)]:
        for nombre, modelo in build_models().items():
            modelo.fit(Xtr, y_train)
            y_pred = modelo.predict(Xva)
            y_proba = modelo.predict_proba(Xva)[:, 1]
            acc = accuracy_score(y_val, y_pred)
            p, r, f1, _ = precision_recall_fscore_support(y_val, y_pred, average='binary')
            filas.append({'Conjunto': etiqueta, 'Modelo': nombre, 'Accuracy': acc,
                          'Precision': p, 'Recall': r, 'F1-Score': f1,
                          'ROC-AUC': roc_auc_score(y_val, y_proba)})

    comp = pd.DataFrame(filas)
    comp.to_csv('docs/figures/metrics_with_negativity.csv', index=False)

    pivote = comp.pivot(index='Modelo', columns='Conjunto', values=['Accuracy', 'F1-Score', 'ROC-AUC'])
    delta = pd.DataFrame({
        'Modelo': pivote.index,
        'Delta_Accuracy': pivote['Accuracy']['Con negatividad'] - pivote['Accuracy']['Sin negatividad'],
        'Delta_F1': pivote['F1-Score']['Con negatividad'] - pivote['F1-Score']['Sin negatividad'],
        'Delta_ROC_AUC': pivote['ROC-AUC']['Con negatividad'] - pivote['ROC-AUC']['Sin negatividad']
    }).reset_index(drop=True)
    delta.to_csv('docs/figures/negativity_delta.csv', index=False)

    print("\nComparacion de modelos con y sin la variable de negatividad:")
    print(comp.to_string(index=False))
    print("\nDiferencia (Con negatividad menos Sin negatividad):")
    print(delta.to_string(index=False))

    plt.figure(figsize=(10, 5))
    sns.barplot(data=comp, x='Modelo', y='F1-Score', hue='Conjunto', palette=['#2b5c8f', '#d9534f'])
    plt.title('F1-Score con y sin la Variable de Negatividad')
    plt.ylim(0.6, 0.85)
    plt.ylabel('F1-Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig('docs/figures/negativity_model_comparison.png', dpi=300)
    plt.close()

    # Poder discriminativo de la negatividad por si sola y sensibilidad al peso
    auc_sola = roc_auc_score(y_val, neg_val.ravel())
    sens = [{'peso': 0.0, 'F1-Score': None, 'ROC-AUC': auc_sola, 'nota': 'negatividad como unico predictor'}]
    for w in [1, 5, 10, 25]:
        Xtr_w = hstack([X_train_vec, csr_matrix(neg_train * w)]).tocsr()
        Xva_w = hstack([X_val_vec, csr_matrix(neg_val * w)]).tocsr()
        m = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(Xtr_w, y_train)
        pred = m.predict(Xva_w)
        _, _, f1, _ = precision_recall_fscore_support(y_val, pred, average='binary')
        sens.append({'peso': w, 'F1-Score': f1,
                     'ROC-AUC': roc_auc_score(y_val, m.predict_proba(Xva_w)[:, 1]),
                     'nota': 'TF-IDF + negatividad escalada (Regresion Logistica)'})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv('docs/figures/negativity_sensitivity.csv', index=False)
    print('\nPoder discriminativo de la negatividad y sensibilidad a su escala:')
    print(sens_df.to_string(index=False))

    return comp, delta


if __name__ == '__main__':
    train_df, test_df = run_part1_eda()
    train_df = run_part2_preprocessing_and_ngrams(train_df)
    res_df = run_part3_models(train_df)
    best_model, tfidf_final = run_part4_classification_function(train_df)
    run_part4_demo(best_model, tfidf_final)
    train_df = run_part5_sentiment(train_df)
    run_part6_sentiment_questions(train_df)
    run_part7_negativity_model(train_df)
    train_df.to_csv('data/train_processed.csv', index=False, encoding='utf-8')
    print(chr(10) + 'Dataset con sentimiento y negatividad guardado en data/train_processed.csv')
