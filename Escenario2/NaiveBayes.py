import os
import re
import warnings
import time
import json
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

warnings.simplefilter("ignore")

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

def classify_documents(ngram_file):
    df = pd.read_csv(ngram_file, names=['ngram', 'identifier', 'label'], delimiter='|', quotechar='"', engine='python')
    df.dropna(inplace=True)
    train_data = df[df['label'] != '???']
    test_data = df[df['label'] == '???']
    le = LabelEncoder()
    y_train_full = le.fit_transform(train_data['label'])
    vectorizer = TfidfVectorizer()
    X_train_full_vect = vectorizer.fit_transform(train_data['ngram'])
    X_test_vect = vectorizer.transform(test_data['ngram'])
    X_train, X_val, y_train, y_val = train_test_split(X_train_full_vect, y_train_full, test_size=0.2, random_state=42)
    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    test_preds = clf.predict(X_test_vect)
    
    predictions = le.inverse_transform(test_preds)
    train_report = classification_report(y_train, train_preds, target_names=le.classes_, output_dict=True)
    val_report = classification_report(y_val, val_preds, target_names=le.classes_, output_dict=True)
    
    results = {
        'Model': 'NaiveBayes',
        'Train Accuracy': accuracy_score(y_train, train_preds),
        'Train Precision': train_report['weighted avg']['precision'],
        'Train Recall': train_report['weighted avg']['recall'],
        'Train F1-Score': train_report['weighted avg']['f1-score'],
        'Val Accuracy': accuracy_score(y_val, val_preds),
        'Val Precision': val_report['weighted avg']['precision'],
        'Val Recall': val_report['weighted avg']['recall'],
        'Val F1-Score': val_report['weighted avg']['f1-score']
    }
    
    validation_txt = ngram_file.replace('.txt', '_validation_results.txt')
    with open(validation_txt, 'w', encoding='utf-8') as out_file:
        json.dump(results, out_file, indent=4)
    
    print(f"Resultados de validación guardados en {validation_txt}")
    print("Resultados de validación:")
    for key, value in results.items():
        print(f"{key}: {value}")
    
if __name__ == "__main__":
    home_dir = os.path.expanduser("~")
    file_name = "corpus_muestreo10-03.txt"
    file_path = os.path.join(home_dir, file_name)
    if os.path.exists(file_path):
        ngram_size = int(input("Elige el tipo de n-grama (1: unigramas, 2: bigramas, 3: trigramas): "))
        if ngram_size in [1, 2, 3]:
            posting_list, filename = process_file(file_path, ngram_size)
            ngram_file = save_results(posting_list, file_path, ngram_size)
            if ngram_file:
                classify_documents(ngram_file)
        else:
            print("Error: Ingresa un número válido (1, 2 o 3).")
