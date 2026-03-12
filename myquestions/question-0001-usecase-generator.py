import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import random


def generar_caso_de_uso_optimizar_y_entrenar():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función optimizar_y_entrenar(df, target_col, k_best=5).

    La función devuelve:
        input_data  : dict con claves 'df', 'target_col' y 'k_best'
        output_data : tupla (modelo_entrenado, columnas_seleccionadas, accuracy)
    """

    # ------------------------------------------------------------------
    # 1. Configuración aleatoria
    # ------------------------------------------------------------------
    n_rows     = random.randint(30, 80)       # filas suficientes para RF
    n_features = random.randint(6, 12)        # más columnas que k_best
    k_best     = random.randint(2, min(5, n_features - 1))

    # ------------------------------------------------------------------
    # 2. Generar DataFrame con ruido
    # ------------------------------------------------------------------
    feature_cols = [f'feature_{i}' for i in range(n_features)]

    # Escalas muy diferentes para simular datos reales de vino
    scales = np.random.uniform(0.1, 10.0, size=n_features)
    data   = np.random.randn(n_rows, n_features) * scales

    df = pd.DataFrame(data, columns=feature_cols)

    # Introducir ~10 % de NaNs
    mask = np.random.choice([True, False], size=df.shape, p=[0.10, 0.90])
    df[mask] = np.nan

    # Target binario: calidad buena (1) / mala (0)
    target_col = 'calidad'
    df[target_col] = np.random.randint(0, 2, size=n_rows)

    # ------------------------------------------------------------------
    # 3. INPUT
    # ------------------------------------------------------------------
    input_data = {
        'df'        : df.copy(),
        'target_col': target_col,
        'k_best'    : k_best,
    }

    # ------------------------------------------------------------------
    # 4. OUTPUT esperado  →  replicamos la lógica de optimizar_y_entrenar
    # ------------------------------------------------------------------

    # A. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # B. Imputar con mediana
    imputer  = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # C. Seleccionar k mejores características
    selector      = SelectKBest(score_func=f_classif, k=k_best)
    X_selected    = selector.fit_transform(X_imputed, y)
    mask_selected = selector.get_support()
    cols_selected = list(X.columns[mask_selected])

    # D. Entrenar Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_selected, y)

    # E. Accuracy inicial (sobre los mismos datos de entrenamiento)
    accuracy = model.score(X_selected, y)

    output_data = (model, cols_selected, accuracy)

    return input_data, output_data


# ----------------------------------------------------------------------
# Ejemplo de uso
# ----------------------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_optimizar_y_entrenar()

    print("=== INPUT ===")
    print(f"  target_col : {entrada['target_col']}")
    print(f"  k_best     : {entrada['k_best']}")
    print(f"  DataFrame shape : {entrada['df'].shape}")
    print("  Primeras filas:")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    modelo, columnas, acc = salida_esperada
    print(f"  Modelo           : {type(modelo).__name__}")
    print(f"  Columnas elegidas: {columnas}")
    print(f"  Accuracy inicial : {acc:.4f}")
