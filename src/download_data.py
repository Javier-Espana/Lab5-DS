import os
import shutil
import urllib.request

def get_dataset():
    """
    Descarga el conjunto de datos de desastres de Kaggle usando kagglehub
    o descarga de respaldo desde repositorio verificado.
    """
    os.makedirs('data', exist_ok=True)
    downloaded = False
    
    # 1. Intentar descarga mediante kagglehub
    try:
        import kagglehub
        print("Intentando descarga con kagglehub...")
        path = kagglehub.competition_download('nlp-getting-started')
        print("Ruta de archivos de la competencia:", path)
        if os.path.exists(path):
            for file_name in os.listdir(path):
                src_file = os.path.join(path, file_name)
                dst_file = os.path.join('data', file_name)
                shutil.copy2(src_file, dst_file)
            print("Archivos copiados exitosamente a la carpeta data/")
            downloaded = True
    except Exception as exc:
        print(f"Descarga con kagglehub omitida o no autenticada: {exc}")
        print("Procediendo con descarga directa de respaldo...")

    # 2. Descarga de respaldo si no esta disponible train.csv
    if not downloaded or not os.path.exists('data/train.csv'):
        urls = {
            'train.csv': 'https://raw.githubusercontent.com/tarunannapareddy/Natural-Language-Processing-with-Disaster-Tweets/main/train.csv',
            'test.csv': 'https://raw.githubusercontent.com/tarunannapareddy/Natural-Language-Processing-with-Disaster-Tweets/main/test.csv'
        }
        for name, url in urls.items():
            dest_path = os.path.join('data', name)
            urllib.request.urlretrieve(url, dest_path)
            print(f"Descargado {name} ({os.path.getsize(dest_path)} bytes)")
            
    print("Contenido final de carpeta data/:", os.listdir('data'))

if __name__ == '__main__':
    get_dataset()
