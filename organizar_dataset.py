from pathlib import Path
import shutil

# ============================
# RUTAS
# ============================

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
    "Coconut": ["cocos"],
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
# RECORRER TRAINING - TEST - VALIDATION
# ============================

for division in ["Training", "Validation", "Test"]:

    carpeta_origen = ORIGEN / division

    carpeta_destino = DESTINO / division

    carpeta_destino.mkdir(parents=True, exist_ok=True)

    print(f"\nProcesando {division}...")

    for carpeta in carpeta_origen.iterdir():

        if not carpeta.is_dir():
            continue

        nombre = carpeta.name.lower()

        fruta_destino = None

        for fruta, palabras in FRUTAS.items():

            for palabra in palabras:

                if palabra in nombre:

                    fruta_destino = fruta
                    break

            if fruta_destino is not None:
                break

        if fruta_destino is None:
            continue
        
        nueva_carpeta = carpeta_destino / fruta_destino

        nueva_carpeta.mkdir(exist_ok=True)

        for imagen in carpeta.iterdir():

            if not imagen.is_file():
                continue

            nuevo_nombre = f"{carpeta.name.replace(' ','_')}_{imagen.name}"

            shutil.copy2(
                imagen,
                nueva_carpeta / nuevo_nombre
            )

print("\n===================================")
print("Dataset organizado correctamente")
print("===================================")
