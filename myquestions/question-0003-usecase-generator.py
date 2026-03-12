import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import random


def generar_caso_de_uso_segmentar_clientes():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función segmentar_clientes(df, n_clusters=3).

    La función devuelve:
        input_data  : dict con claves 'df' y 'n_clusters'
        output_data : tupla (df_modificado, centroides)
    """

    # ------------------------------------------------------------------
    # 1. Configuración aleatoria
    # ------------------------------------------------------------------
    n_rows      = random.randint(30, 100)
    n_clusters  = random.randint(2, 5)

    # Columnas numéricas que simulan datos de e-commerce
    col_opciones = [
        'gasto_total', 'frecuencia_compra', 'ticket_promedio',
        'dias_desde_ultima_compra', 'num_productos_distintos'
    ]
    n_cols        = random.randint(2, len(col_opciones))
    cols_elegidas = random.sample(col_opciones, n_cols)

    # ------------------------------------------------------------------
    # 2. Generar DataFrame con datos de clientes
    # ------------------------------------------------------------------
    data = {}
    for col in cols_elegidas:
        if col == 'gasto_total':
            data[col] = np.random.uniform(10, 5000, size=n_rows)
        elif col == 'frecuencia_compra':
            data[col] = np.random.randint(1, 50, size=n_rows).astype(float)
        elif col == 'ticket_promedio':
            data[col] = np.random.uniform(5, 500, size=n_rows)
        elif col == 'dias_desde_ultima_compra':
            data[col] = np.random.randint(1, 365, size=n_rows).astype(float)
        elif col == 'num_productos_distintos':
            data[col] = np.random.randint(1, 30, size=n_rows).astype(float)

    # Añadir columna no numérica para que la función deba filtrar
    data['nombre_cliente'] = [f'cliente_{i}' for i in range(n_rows)]

    df = pd.DataFrame(data)

    # Introducir ~10% de NaNs solo en columnas numéricas
    for col in cols_elegidas:
        mask = np.random.choice([True, False], size=n_rows, p=[0.10, 0.90])
        df.loc[mask, col] = np.nan

    # ------------------------------------------------------------------
    # 3. INPUT
    # ------------------------------------------------------------------
    input_data = {
        'df'        : df.copy(),
        'n_clusters': n_clusters,
    }

    # ------------------------------------------------------------------
    # 4. OUTPUT esperado  →  replicamos la lógica de segmentar_clientes
    # ------------------------------------------------------------------
    df_out = df.copy()

    # A. Extraer columnas numéricas
    X = df_out.select_dtypes(include=[np.number])

    # B. Imputar con constante 0
    imputer   = SimpleImputer(strategy='constant', fill_value=0)
    X_imputed = imputer.fit_transform(X)

    # C. K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_imputed)

    # D. Añadir columna segmento_id
    df_out['segmento_id'] = kmeans.labels_

    # E. Centroides
    centroides = kmeans.cluster_centers_

    output_data = (df_out, centroides)

    return input_data, output_data


# ----------------------------------------------------------------------
# Ejemplo de uso
# ----------------------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_segmentar_clientes()

    print("=== INPUT ===")
    print(f"  n_clusters     : {entrada['n_clusters']}")
    print(f"  DataFrame shape: {entrada['df'].shape}")
    print(f"  Columnas       : {list(entrada['df'].columns)}")
    print("  Primeras filas:")
    print(entrada['df'].head())

    print("\n=== OUTPUT ESPERADO ===")
    df_resultado, centroides = salida_esperada
    print(f"  DataFrame shape (con segmento_id): {df_resultado.shape}")
    print(f"  Distribución de segmentos:\n{df_resultado['segmento_id'].value_counts()}")
    print(f"\n  Centroides shape : {centroides.shape}")
    print(f"  Centroides:\n{centroides}")
