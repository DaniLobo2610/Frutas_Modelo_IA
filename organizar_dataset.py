import random
import re
import shutil
from pathlib import Path

random.seed(42)

# ==========================================
# CONFIGURACIÓN
# ==========================================

MAX_ORIGINALES = 2000
MAX_MULTI = 300

ORIGEN = Path("dataset_original")
MULTI = Path("fruits-360_multi/test-multiple_fruits")
DESTINO = Path("dataset")

FRUTAS = {
    "Apple": ["apple"],
    "Avocado": ["avocado"],
    "Banana": ["banana"],
    "Blackberry": ["blackberry"],
    "Cactus Fruit": ["cactus fruit"],
    "Carambola": ["carambola"],
    "Cherry": ["cherry"],
    "Grape": ["grape"],
    "Orange": ["orange"],
    "Papaya": ["papaya"],
    "Peach": ["peach"],
    "Pear": ["pear"],
    "Raspberry": ["raspberry"],
    "Strawberry": ["strawberry"],
    "Tomato": ["tomato"]
}

def imagen_valida(nombre_archivo, palabra):

    nombre = nombre_archivo.lower()

    patron = rf"^{re.escape(palabra.replace(' ', '_'))}_[0-9]+$"

    return re.match(patron, nombre) is not None


for division in ["Training","Validation","Test"]:

    print(f"\n{'='*50}")
    print(f"Procesando {division}")
    print(f"{'='*50}")

    carpeta_origen = ORIGEN / division
    carpeta_destino = DESTINO / division

    carpeta_destino.mkdir(parents=True, exist_ok=True)

    resumen = {}

    for fruta, palabras in FRUTAS.items():

        destino_fruta = carpeta_destino / fruta
        destino_fruta.mkdir(exist_ok=True)

        imagenes = []

        for carpeta in carpeta_origen.iterdir():

            if not carpeta.is_dir():
                continue

            nombre = carpeta.name.lower()

            if any(p in nombre for p in palabras):

                for img in carpeta.iterdir():

                    if img.is_file():

                        imagenes.append((carpeta.name,img))

        random.shuffle(imagenes)

        imagenes = imagenes[:MAX_ORIGINALES]

        contador = 0

        for carpeta_nombre,img in imagenes:

            nuevo = f"{carpeta_nombre.replace(' ','_')}_{img.name}"

            shutil.copy2(

                img,

                destino_fruta / nuevo

            )

            contador += 1

        if division == "Training":

            multi = []

            for img in MULTI.iterdir():

                if not img.is_file():
                    continue

                for palabra in palabras:

                    if imagen_valida(img.stem,palabra):

                        multi.append(img)

                        break

            random.shuffle(multi)

            multi = multi[:MAX_MULTI]

            for i,img in enumerate(multi):

                nuevo = f"multi_{i+1}{img.suffix}"

                shutil.copy2(

                    img,

                    destino_fruta / nuevo

                )

                contador += 1

        resumen[fruta]=contador


    print("\nResumen:\n")

    for fruta,cantidad in resumen.items():

        print(f"{fruta:<18} -> {cantidad} imágenes")

print("\n===================================")
print("Dataset organizado correctamente")
print("===================================")