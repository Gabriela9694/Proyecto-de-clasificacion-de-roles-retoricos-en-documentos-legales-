import pandas as pd
import os
import nltk
import time
from nltk.tokenize import word_tokenize
import re
import string

nltk.download('punkt')
nltk.download('punkt_tab')
# Configuraciones de rutas y parámetros
ARCHIVO_STOPWORDS = r"/home2/darnes/tarea6/stopwords2.txt"
PALABRAS_ELIMINAR = {
    "aa", "xxx", "aba", "ab", "abc", "zrzu", "jazg", "jaiį",
    "zsu", "pgt", "ouu", "zsz", "dgĥrz", "aiʈaiī", "zulfikar", "ali", "pakistan",
    "zõz", "zsu", "pgt", "zɣv", "ivrwzg", "dzg", "zɣvzê", "jazg", "pamp",
    "îzgju", "aiiz", "palĩp", "îzgju", "palĩp", "qīzi", "ïvz", "uߪzu", "pýz",
    "ðøju", "paam", "pqt", "üwzg", "ezu", "ߣau", "ýpaiģ", "pɼvav", "zrvz",
    "ýpaiģ", "wvz", "aiģ", "ýzg", "czpv", "qwzã", "ýzu", "ðøju", "paam",
    "īig", "qav", "ev", "ĺav", "wjpaq", "vaz", "ɬaz", "vzu", "pľv", "gÄ³zu",
    "ɸju", "zrzu", "jazg", "ʸvp", "pu", "cfaiģ", "ߣau", "plzv", "ivqīz",
    "wåu", "uĥz", "wåu", "aaz", "gparz", "aaz", "aazav", "czgav", "qai", "aazav", "zaz", "aazlav", "smuļ", "itv", "aazsz", 
    "maz", "du", "wu", "waiģ", "jutz", "wzu", "ejpÆ½", "eyk", "ezg", "ezgai", "ezp", "fao",
    "zjaiīg", "gju", "xaivai", "cfu", "eirgvg","zgu", "iqz", "wu", "zjaiīg", "ez", "jaz", "gju", "wīav", "poraiģ", "nan",
    "jaiÄ«z","iwaiÄ£","qaiįvz", "iwaiģ", "cxirpaq", "agīzjaz", "irzÃ£", "iqÄ«z", "cfai", "cfto", "cgo", "cue","cuÉ¢z",
    "qu", "ju", "vawp", "ja", "aagÄ«Å£", "aan", "aarya", "adc","ade", "agvÃ£", "agwg","agÄ«zjaz", "ahs", "aiÄ«Ä«Ã£",
    "irz", "irzgai", "irzÃ£", "isak", "iscan", "iz", "izj", "jaiÄ«z", "jzaiÄ£", "maa", "ayb", "ayengar", "ays", "ayurvedic", "azad", 
    "azai", "azg", "azlz", "azu", "azÃ£", "Ãªu"
}

NUMEROS_ROMANOS_REGEX = re.compile(r'^(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$', re.IGNORECASE)
NUMEROS_ROMANOS_MINUSCULAS_REGEX = re.compile(r'^(m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3}))$')

CARACTERES_INVALIDOS_REGEX = re.compile(r'[^\x00-\x7F]')  # Regex para caracteres no ASCII


def contiene_caracteres_no_ascii(palabra):
    return bool(CARACTERES_INVALIDOS_REGEX.search(palabra))


def cargar_stopwords():
    try:
        if not os.path.exists(ARCHIVO_STOPWORDS):
            raise FileNotFoundError(f"No se encontró el archivo de stopwords en '{ARCHIVO_STOPWORDS}'")
        
        with open(ARCHIVO_STOPWORDS, 'r', encoding='utf-8') as f:
            return {line.strip().lower() for line in f if line.strip()}
    except Exception as e:
        print(f"Error al cargar stopwords: {e}")
        return set()


def procesar_texto(texto, stop_words):
    if not isinstance(texto, str):
        return ""
    
    palabras = word_tokenize(texto)
    palabras_procesadas = []
    
    for palabra in palabras:
        palabra = palabra.lower()
        
        if (contiene_caracteres_no_ascii(palabra) or
            palabra.isdigit() or
            len(palabra) == 1 or
            (len(palabra) == 2 and not any(vocal in palabra for vocal in 'aeiouáéíóú')) or
            (len(palabra) > 1 and not any(vocal in palabra for vocal in 'aeiouáéíóú')) or
            re.search(r'\d+[a-zA-Z]|[a-zA-Z]+\d', palabra) or
            re.fullmatch(NUMEROS_ROMANOS_REGEX, palabra) or
            re.fullmatch(NUMEROS_ROMANOS_MINUSCULAS_REGEX, palabra) or
            re.search(r'([bcdfghjklmnpqrstvwxyz])\1{1,}', palabra) or
            re.fullmatch(r'(\w)\1{4,}', palabra) or
            re.search(r'[^\w\s]', palabra) or
            palabra in PALABRAS_ELIMINAR):
            continue
        
        if palabra not in stop_words and palabra not in string.punctuation:
            palabras_procesadas.append(palabra)
    
    return ' '.join(palabras_procesadas)


def leer_archivo(input_path):
    try:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
        
        ext = os.path.splitext(input_path)[1].lower()
        
        if ext == ".csv":
            return pd.read_csv(input_path, header=None, names=['id', 'texto', 'label'])
        elif ext == ".txt":
            with open(input_path, 'r', encoding='utf-8') as f:
                lineas = [line.strip().split(',', 2) for line in f if line.strip()]
                return pd.DataFrame(lineas, columns=['id', 'texto', 'label'])
        else:
            raise ValueError("Formato de archivo no soportado. Use CSV o TXT.")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return None


def procesar_archivo(input_path, output_folder):
    stop_words = cargar_stopwords()
    if not stop_words:
        print("No se pueden procesar los textos sin las stopwords")
        return
    
    data = leer_archivo(input_path)
    if data is None:
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    archivos_generados = 0
    total_archivos = len(data)
    start_time = time.time()
    
    for index, row in data.iterrows():
        try:
            if row.isnull().all():
                continue
            
            texto_procesado = procesar_texto(str(row['texto']), stop_words)
            
            if not texto_procesado.strip():
                continue
                
            contenido = (
                f"ID: {row['id']}\n"
                f"TEXTO PROCESADO: {texto_procesado}\n"
                f"ETIQUETA: {row['label']}\n"
            )
            
            nombre_archivo = f"{row['id']}.txt"
            ruta_archivo = os.path.join(output_folder, nombre_archivo)
            
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write(contenido)
                archivos_generados += 1
            
            print(f"Procesando archivo {archivos_generados}/{total_archivos}")
        except Exception as e:
            print(f"Error en fila {index + 1}: {e}")
    
    elapsed_time = time.time() - start_time
    print(f"Procesamiento completado en {elapsed_time:.2f} segundos. Archivos generados: {archivos_generados}")
    print(f"Ubicación de los archivos: {os.path.abspath(output_folder)}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    input_file = os.path.join(script_dir, "corpus.txt")  # Se puede cambiar a un archivo TXT
    output_folder = os.path.join(script_dir, "corpus_procesado")
    
    procesar_archivo(input_file, output_folder)


if __name__ == "__main__":
    main()
