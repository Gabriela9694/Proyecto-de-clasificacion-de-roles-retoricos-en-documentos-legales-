import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler  # Para asegurar valores no negativos
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import nltk
import warnings

# Ignorar warnings específicos de métricas
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")

nltk.download('punkt')
nltk.download('stopwords')

# Lectura y procesamiento de datos
def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line_number, line in enumerate(f, start=1):  # Contador de líneas para depuración
            parts = line.strip().split(',', 1)  # Dividir por la primera coma
            if len(parts) < 2:  # Validar que haya al menos dos partes
                print(f"Advertencia: Línea {line_number} ignorada por formato incorrecto: '{line.strip()}'")
                continue  # Saltar esta línea
            
            doc_id = parts[0]
            rest = parts[1].rsplit(',', 1)  # Dividir por la última coma
            if len(rest) < 2:  # Validar que haya texto y etiqueta
                print(f"Advertencia: Línea {line_number} ignorada por formato incorrecto: '{line.strip()}'")
                continue  # Saltar esta línea
            
            text = rest[0]
            label = rest[1]
            data.append((doc_id, text, label))
    
    return data

# Preprocesamiento
def preprocess(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    return ' '.join(tokens)

# Carga de datos
data = load_data('corpus_procesado10-03.txt')

# Separar datos en entrenamiento y prueba basados en la presencia de '???'
train_data = [d for d in data if d[2] != '???']
test_data = [d for d in data if d[2] == '???']

# Verificar si hay datos de prueba
has_test_data = len(test_data) > 0

# Preparación de datasets
if len(train_data) == 0:
    raise ValueError("No se encontraron datos para entrenamiento. Asegúrate de que el corpus contenga etiquetas válidas.")

X = [preprocess(text) for _, text, _ in train_data]
y = [label for _, _, label in train_data]

# División de datos en entrenamiento (80%) y validación (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline combinado LSA + LDA con transformación para eliminar valores negativos
combined_features = FeatureUnion([
    ('lsa', Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svd', TruncatedSVD()),
        ('scaler', MinMaxScaler())  # Asegura valores no negativos
    ])),
    ('lda', Pipeline([
        ('count', CountVectorizer()),
        ('lda', LatentDirichletAllocation())
    ]))
])

pipeline = Pipeline([
    ('features', combined_features),
    ('clf', MultinomialNB())  # Usamos MultinomialNB
])

# Parámetros para Grid Search
params = {
    'features__lsa__tfidf__max_df': [0.75, 1.0],
    'features__lsa__svd__n_components': [100, 200],
    'features__lda__count__max_df': [0.75, 1.0],
    'features__lda__lda__n_components': [50, 100],
    'clf__alpha': [0.1, 0.5, 1.0],  # Parámetro de suavizado para MultinomialNB
}

# Definir funciones de puntuación personalizadas
precision_scorer = make_scorer(precision_score, average='macro', zero_division=0)
recall_scorer = make_scorer(recall_score, average='macro', zero_division=0)
f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)

# Configuración de métricas para GridSearch
scoring = {
    'accuracy': 'accuracy',
    'precision': precision_scorer,
    'recall': recall_scorer,
    'f1': f1_scorer
}

# Búsqueda de hiperparámetros con validación cruzada
grid = GridSearchCV(
    pipeline,
    params,
    cv=3,
    n_jobs=-1,
    verbose=1,
    scoring=scoring,
    refit='f1',
    return_train_score=True
)
grid.fit(X_train, y_train)

# Nombre del algoritmo
algorithm_name = "MultinomialNB"

# Métricas promedio durante validación cruzada
cv_results = grid.cv_results_
best_index = grid.best_index_

print(f"\n--- Métricas Promedio (Validación Cruzada) - {algorithm_name} ---")
print(f"{'':<15} | {'Entrenamiento':<15} | {'Validación':<15}")
print(f"{'Accuracy':<15} | {cv_results['mean_train_accuracy'][best_index]:.4f} | {cv_results['mean_test_accuracy'][best_index]:.4f}")
print(f"{'Precision (macro)':<15} | {cv_results['mean_train_precision'][best_index]:.4f} | {cv_results['mean_test_precision'][best_index]:.4f}")
print(f"{'Recall (macro)':<15} | {cv_results['mean_train_recall'][best_index]:.4f} | {cv_results['mean_test_recall'][best_index]:.4f}")
print(f"{'F1-score (macro)':<15} | {cv_results['mean_train_f1'][best_index]:.4f} | {cv_results['mean_test_f1'][best_index]:.4f}")

# Evaluación en el conjunto de validación
best_model = grid.best_estimator_
y_val_pred = best_model.predict(X_val)

# Métricas en entrenamiento completo
y_train_pred = best_model.predict(X_train)

# Resumen de métricas
def print_metrics_summary(y_true, y_pred, dataset_name, algorithm_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)  # Controla el comportamiento
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)        # Controla el comportamiento
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)                # Controla el comportamiento
    
    print(f"\n--- Métricas en {dataset_name} ({algorithm_name}) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

# Imprimir métricas para entrenamiento y validación
print_metrics_summary(y_train, y_train_pred, "Entrenamiento", algorithm_name)
print_metrics_summary(y_val, y_val_pred, "Validación", algorithm_name)

# Informe de clasificación detallado
print("\n--- Informe de Clasificación (Validación) ---")
print(classification_report(y_val, y_val_pred, zero_division=0))

# Predicciones en datos de prueba (si existen)
if has_test_data:
    X_test = [preprocess(text) for _, text, _ in test_data]
    test_ids = [doc_id for doc_id, _, _ in test_data]
    test_predictions = best_model.predict(X_test)

    # Creación de archivo de resultados en formato TXT
    with open('naive_bayes_classifications.txt', 'w') as f:
        f.write("=== Resultados de Clasificación ===\n\n")
        f.write(f"Total datos entrenamiento: {len(train_data)}\n")
        f.write(f"Total datos validación: {len(X_val)}\n")
        f.write(f"Total datos clasificación: {len(test_data)}\n\n")
        f.write("Predicciones:\n")
        for doc_id, text, label in zip(test_ids, [text for _, text, _ in test_data], test_predictions):
            f.write(f"ID: {doc_id}, Texto: {text}, Etiqueta Predicha: {label}\n")
    print("\nArchivo 'naive_bayes_classifications.txt' generado con las predicciones.")
else:
    print("\nNo se encontraron datos de prueba con etiqueta '???'. No se generó el archivo TXT.")

# Resumen final
print("\n--- Resumen Final ---")
print(f"Total datos entrenamiento: {len(train_data)}")
print(f"Total datos validación: {len(X_val)}")
print(f"Total datos clasificación: {len(test_data)}")