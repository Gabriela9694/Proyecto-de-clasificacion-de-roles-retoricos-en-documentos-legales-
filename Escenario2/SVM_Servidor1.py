import os
import re
import warnings
import time
import argparse
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

warnings.simplefilter("ignore", category=FutureWarning)

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def generate_ngrams(words, n):
    if len(words) < n:
        return []
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def process_line(line, ngram_size):
    posting_list = defaultdict(set)
    parts = line.strip().split(',', 2)
    if len(parts) < 3:
        return posting_list
    identifier, text, label = parts
    words = tokenize(text)
    ngrams = generate_ngrams(words, ngram_size)
    for ngram in ngrams:
        posting_list[ngram].add((identifier, label))
    return posting_list

def process_file(file_path, ngram_size, chunksize=1000):
    posting_list = defaultdict(set)
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    
    total_lines = len(lines)
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(process_line, line, ngram_size): line for line in lines}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            for key, value in result.items():
                posting_list[key].update(value)
            if (i + 1) % chunksize == 0 or (i + 1) == total_lines:
                elapsed_time = time.time() - start_time
                print(f"Procesadas {i+1}/{total_lines} líneas en {elapsed_time:.2f} segundos")
    
    return posting_list, file_path

def save_results(posting_list, file_path, ngram_size):
    output_path = file_path.replace('.txt', f'_posting_list_{ngram_size}gram.txt')
    with open(output_path, 'w', encoding='utf-8') as out_file:
        for ngram, entries in posting_list.items():
            for identifier, label in entries:
                out_file.write(f'"{ngram}"|"{identifier}"|"{label}"\n')
    return output_path

def classify_documents(ngram_file, use_smote=False, fast_mode=True):
    df = pd.read_csv(ngram_file, names=['ngram', 'identifier', 'label'], delimiter='|', quotechar='"', engine='python')
    df.dropna(inplace=True)

    train_data = df[df['label'] != '???']
    test_data = df[df['label'] == '???']

    # Reducir muestra para pruebas rápidas
    if fast_mode and len(train_data) > 100000:
        train_data = train_data.sample(frac=0.3, random_state=42)

    le = LabelEncoder()
    y_train_full = le.fit_transform(train_data['label'])

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), use_idf=True, sublinear_tf=True)
    X_train_full_vect = vectorizer.fit_transform(train_data['ngram'])
    X_test_vect = vectorizer.transform(test_data['ngram'])

    X_train, X_val, y_train, y_val = train_test_split(X_train_full_vect, y_train_full, test_size=0.2, random_state=42)

    if use_smote and len(y_train) < 50000:  # Evitar SMOTE en grandes datasets
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

    scaler = MaxAbsScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test_vect = scaler.transform(X_test_vect)

    # Usar SVC con kernel lineal para mejorar velocidad
    clf = SVC(kernel="linear", class_weight="balanced", probability=True)
    clf.fit(X_train, y_train)

    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test_vect)

    predictions = le.inverse_transform(test_preds)
    train_report = classification_report(y_train, train_preds, target_names=le.classes_, output_dict=True)
    val_report = classification_report(y_val, val_preds, target_names=le.classes_, output_dict=True)

    results = (
        f"Model: SVM (kernel=linear)\n"
        f"Train Accuracy: {accuracy_score(y_train, train_preds)}\n"
        f"Train Precision: {train_report['weighted avg']['precision']}\n"
        f"Train Recall: {train_report['weighted avg']['recall']}\n"
        f"Train F1-Score: {train_report['weighted avg']['f1-score']}\n"
        f"Val Accuracy: {accuracy_score(y_val, val_preds)}\n"
        f"Val Precision: {val_report['weighted avg']['precision']}\n"
        f"Val Recall: {val_report['weighted avg']['recall']}\n"
        f"Val F1-Score: {val_report['weighted avg']['f1-score']}\n"
    )

    validation_txt = ngram_file.replace('.txt', '_validation_results.txt')
    with open(validation_txt, 'w', encoding='utf-8') as out_file:
        out_file.write(results)

    print(f"Resultados de validación guardados en {validation_txt}")
    print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="corpus_muestreo10-03.txt")
    parser.add_argument("--ngram", type=int, choices=[1, 2, 3], required=True, help="Tipo de n-grama: 1=unigramas, 2=bigramas, 3=trigramas")
    parser.add_argument("--fast", action="store_true", help="Modo rápido para grandes datasets")
    args = parser.parse_args()

    if os.path.exists(args.file):
        posting_list, filename = process_file(args.file, args.ngram)
        ngram_file = save_results(posting_list, args.file, args.ngram)
        if ngram_file:
            classify_documents(ngram_file, fast_mode=args.fast)
    else:
        print("Error: Archivo no encontrado.")
