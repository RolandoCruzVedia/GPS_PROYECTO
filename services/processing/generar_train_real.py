"""
processing/generar_train_real.py

Genera dataset_train_real.csv usando EXACTAMENTE el mismo split
train/test que generar_sinteticos.py (mismo SEED, mismo FRAC_TEST),
pero SIN agregar filas sinteticas. Sirve para evaluar el modelo
solo con datos reales, antes de compararlo contra la version aumentada.
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
IN_FILE = BASE_DIR / "dataset.csv"
OUT_TRAIN_REAL_FILE = BASE_DIR / "dataset_train_real.csv"

FRAC_TEST = 0.2
SEED = 42


def main():
    df_real = pd.read_csv(IN_FILE)
    df_real["es_sintetico"] = 0

    df_test = df_real.sample(frac=FRAC_TEST, random_state=SEED)
    df_train_real = df_real.drop(df_test.index)

    df_train_real.to_csv(OUT_TRAIN_REAL_FILE, index=False)

    print(f"[OK] Train real (sin sinteticos): {OUT_TRAIN_REAL_FILE} ({len(df_train_real)} filas)")
    print(f"[INFO] Este split usa el mismo SEED={SEED} y FRAC_TEST={FRAC_TEST} "
          f"que generar_sinteticos.py, por lo que dataset_test_real.csv "
          f"sigue siendo valido para comparar ambos escenarios.")


if __name__ == "__main__":
    main()
