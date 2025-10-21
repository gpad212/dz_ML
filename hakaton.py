import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ===============================
# Настройки
# ===============================

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

# ===============================
# Загрузка данных
# ===============================

print("Загрузка данных...")
df = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

print(f"Размер train: {df.shape}, test: {df_test.shape}")

# ===============================
# Предобработка
# ===============================

print("Предобработка текста...")

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

tokenizer = RegexpTokenizer(r'[A-Za-z]+')
wnl = WordNetLemmatizer()

def clean_and_lemmatize(text):
    tokens = tokenizer.tokenize(str(text))
    lemmas = [wnl.lemmatize(word.lower()) for word in tokens]
    return ' '.join(lemmas)

df['clean_url'] = df['url'].apply(clean_and_lemmatize)
df_test['clean_url'] = df_test['url'].apply(clean_and_lemmatize)

# ===============================
# Векторизация (TF-IDF)
# ===============================

print("Построение TF-IDF признаков...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
X = vectorizer.fit_transform(df['clean_url'])
X_test = vectorizer.transform(df_test['clean_url'])
y = df['Predicted']

# ===============================
# Разделение на обучающую и валидационную части
# ===============================

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ===============================
# Обучение моделей
# ===============================

def evaluate_model(model, name):
    model.fit(X_train, y_train)
    pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, pred)
    prec = precision_score(y_valid, pred)
    rec = recall_score(y_valid, pred)
    print(f"{name}: accuracy={acc:.4f}, precision={prec:.4f}, recall={rec:.4f}")
    return model

print("\nОбучение моделей...")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

trained_models = {}
for name, model in models.items():
    trained_models[name] = evaluate_model(model, name)

# ===============================
# Предсказания и сохранение результатов
# ===============================

def save_predictions(model, name):
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        "Id": np.arange(len(preds)),
        "Predicted": preds
    })
    file_name = f"sample_submit_{name.lower()}.csv"
    submission.to_csv(file_name, index=False)
    print(f"Файл {file_name} сохранен.")

print("\nСоздание файлов предсказаний...")
for name, model in trained_models.items():
    save_predictions(model, name)

print("\n✅ Все модели обучены и результаты сохранены.")
