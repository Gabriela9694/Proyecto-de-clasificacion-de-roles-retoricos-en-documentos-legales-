import csv
import time
import concurrent.futures
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Configuración global
INPUT_FILE = "corpus_balanceado.txt"
OUTPUT_FILE = "resultado2.csv"
RANDOM_SEED = 42
THREADS = 6
TEST_SIZE = 0.2

class ClasificadorMultihilo:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.modelos = {
            'J48': DecisionTreeClassifier(random_state=RANDOM_SEED),
            'RandomForest': RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED),
            'Perceptron': Perceptron(n_jobs=-1, random_state=RANDOM_SEED),
            'SVM': SVC(kernel='linear', random_state=RANDOM_SEED),
            'NaiveBayes': MultinomialNB(),
            'RandomTree': DecisionTreeClassifier(
                random_state=RANDOM_SEED,
                max_features='sqrt',
                splitter='random'
            )
        }
    
    def cargar_datos(self):
        try:
            conocidos = []
            desconocidos = []
            with open(INPUT_FILE, 'r', encoding='utf-8') as f:
                lector = csv.reader(f)
                for linea, fila in enumerate(lector, 1):
                    try:
                        if len(fila) != 3:
                            continue
                        doc_id, texto, etiqueta = fila
                        if etiqueta == '???':
                            desconocidos.append((doc_id, texto))
                        else:
                            conocidos.append((texto, etiqueta))
                    except Exception as e:
                        print(f"Error línea {linea}: {str(e)}")
            return conocidos, desconocidos
        except Exception as e:
            print(f"Error al cargar archivo: {str(e)}")
            exit(1)
    
    def preprocesar(self, textos):
        return self.vectorizer.fit_transform(textos)
    
    def calcular_metricas(self, y_true, y_pred, nombre, tipo):
        # Calculamos métricas extendidas
        precision = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        print(f"\n[{nombre}] Métricas {tipo}:")
        print(f"- Exactitud (Accuracy): {precision:.4f}")
        print(f"- Precisión Promedio: {report['macro avg']['precision']:.4f}")
        print(f"- Recall Promedio: {report['macro avg']['recall']:.4f}")
        print(f"- F1-Score Promedio: {report['macro avg']['f1-score']:.4f}")
        print(f"- Matriz de Confusión:\n{confusion_matrix(y_true, y_pred)}")
    
    def entrenar_evaluar_modelo(self, nombre, modelo, X_train, X_test, y_train, y_test):
        try:
            inicio = time.time()
            modelo.fit(X_train, y_train)
            tiempo = time.time() - inicio
            
            # Predicciones y métricas
            train_pred = modelo.predict(X_train)
            test_pred = modelo.predict(X_test)
            
            # Métricas entrenamiento
            print(f"\n{'='*40}\n--- {nombre} ---\n{'='*40}")
            self.calcular_metricas(y_train, train_pred, nombre, "ENTRENAMIENTO")
            
            # Métricas prueba
            self.calcular_metricas(y_test, test_pred, nombre, "PRUEBA")
            print(f"\nTiempo total entrenamiento: {tiempo:.2f} segundos")
            
            return modelo
        except Exception as e:
            print(f"Error en {nombre}: {str(e)}")
            return None
    
    def procesar_desconocidos(self, modelos, datos_desconocidos):
        try:
            textos = [texto for _, texto in datos_desconocidos]
            X = self.vectorizer.transform(textos)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
                futures = {
                    executor.submit(modelo.predict, X): nombre
                    for nombre, modelo in modelos.items()
                }
                resultados = {}
                for future in concurrent.futures.as_completed(futures):
                    nombre = futures[future]
                    resultados[nombre] = future.result()
            
            lineas_resultado = []
            for i, (doc_id, texto) in enumerate(datos_desconocidos):
                fila = [doc_id, texto, '???']
                for nombre in modelos:
                    fila.append(resultados[nombre][i])
                lineas_resultado.append(fila)
            
            return lineas_resultado
        except Exception as e:
            print(f"Error procesando desconocidos: {str(e)}")
            return []

def main():
    inicio_total = time.time()
    clasificador = ClasificadorMultihilo()
    
    try:
        datos_conocidos, datos_desconocidos = clasificador.cargar_datos()
        print(f"\nEstadísticas:")
        print(f"- Registros etiquetados: {len(datos_conocidos)}")
        print(f"- Registros a predecir: {len(datos_desconocidos)}")
        
        if not datos_conocidos:
            print("Error: No hay datos de entrenamiento")
            return
        
        textos, etiquetas = zip(*datos_conocidos)
        X = clasificador.preprocesar(textos)
        y = np.array(etiquetas)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
        )
        print(f"\nDivisión de datos:")
        print(f"- Entrenamiento: {X_train.shape[0]} muestras")
        print(f"- Prueba: {X_test.shape[0]} muestras")
        
        modelos_entrenados = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=THREADS) as executor:
            futures = []
            for nombre, modelo in clasificador.modelos.items():
                futures.append(
                    executor.submit(
                        clasificador.entrenar_evaluar_modelo,
                        nombre, modelo, X_train, X_test, y_train, y_test
                    )
                )
            
            for future in concurrent.futures.as_completed(futures):
                modelo = future.result()
                if modelo:
                    modelos_entrenados[type(modelo).__name__] = modelo
        
        if datos_desconocidos:
            resultados = clasificador.procesar_desconocidos(clasificador.modelos, datos_desconocidos)
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
                escritor = csv.writer(f)
                encabezado = ['Documento', 'Texto', 'Etiqueta_Original'] + list(clasificador.modelos.keys())
                escritor.writerow(encabezado)
                escritor.writerows(resultados)
            print(f"\nResultados guardados en {OUTPUT_FILE}")
        
        print(f"\nTiempo total ejecución: {time.time() - inicio_total:.2f} segundos")
    
    except Exception as e:
        print(f"Error general: {str(e)}")

if __name__ == "__main__":
    main()