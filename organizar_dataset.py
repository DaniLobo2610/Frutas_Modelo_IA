import random
import shutil
from pathlib import Path

random.seed(42)

# ============================
# CONFIGURACIÓN
# ============================

MAX_IMAGENES = 2000

ORIGEN = Path("dataset_original")
DESTINO = Path("dataset")

# ============================
# FRUTAS A UTILIZAR
# ============================

FRUTAS = {
    "Apple": ["apple"],
    "Avocado": ["avocado"],
    "Banana": ["banana"],
    "Blackberry": ["blackberry"],
    "Carambola": ["carambola"],
    "Cherry": ["cherry"],
    "Cactus Fruit": [
    "cactus fruit"],
    "Grape": ["grape"],
    "Orange": ["orange"],
    "Papaya": ["papaya"],
    "Peach": ["peach"],
    "Pear": ["pear"],
    "Raspberry": ["raspberry"],
    "Strawberry": ["strawberry"],
    "Tomato": ["tomato"]
}

# ============================
# RECORRER TRAINING - VALIDATION - TEST
# ============================

for division in ["Training", "Validation", "Test"]:

    print(f"\n==============================")
    print(f"Procesando {division}")
    print(f"==============================")

    carpeta_origen = ORIGEN / division
    carpeta_destino = DESTINO / division

    carpeta_destino.mkdir(parents=True, exist_ok=True)

    resumen = {}

    for fruta in FRUTAS.keys():

        nueva_carpeta = carpeta_destino / fruta
        nueva_carpeta.mkdir(exist_ok=True)

        imagenes = []

        # Buscar todas las carpetas que pertenecen a esa fruta
        for carpeta in carpeta_origen.iterdir():

            if not carpeta.is_dir():
                continue

            nombre = carpeta.name.lower()

            if any(palabra in nombre for palabra in FRUTAS[fruta]):

                for imagen in carpeta.iterdir():

                    if imagen.is_file():

                        imagenes.append((carpeta.name, imagen))

        # Mezclar las imágenes
        random.shuffle(imagenes)

        # Limitar a MAX_IMAGENES
        if len(imagenes) > MAX_IMAGENES:

            imagenes = imagenes[:MAX_IMAGENES]

        # Copiar imágenes
        contador = 0

        for nombre_carpeta, imagen in imagenes:

            nuevo_nombre = f"{nombre_carpeta.replace(' ','_')}_{imagen.name}"

            shutil.copy2(

                imagen,

                nueva_carpeta / nuevo_nombre

            )

            contador += 1

        resumen[fruta] = contador

    print("\nResumen:")

    for fruta, cantidad in resumen.items():

        print(f"{fruta:<15} -> {cantidad:>5} imágenes")

print("\n===================================")
print("Dataset organizado correctamente")
print("===================================")