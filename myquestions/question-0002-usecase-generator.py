import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import random


def generar_caso_de_uso_reducir_y_clasificar():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función reducir_y_clasificar(X, y, varianza_deseada=0.95).

    La función devuelve:
        input_data  : dict con claves 'X', 'y' y 'varianza_deseada'
        output_data : tupla (n_componentes, modelo_entrenado)
    """

    # ------------------------------------------------------------------
    # 1. Configuración aleatoria
    # ------------------------------------------------------------------
    n_rows       = random.randint(50, 120)      # muestras suficientes para KNN
    n_sensors    = random.randint(10, 50)       # simula los sensores de la fábrica
    varianza     = round(random.uniform(0.80, 0.99), 2)  # % de varianza deseada
    n_clases     = random.randint(2, 4)

    # ------------------------------------------------------------------
    # 2. Generar datos con correlación entre sensores (realista)
    # ------------------------------------------------------------------
    # Creamos una base de señales latentes y las mezclamos para simular correlación
    n_latentes = max(3, n_sensors // 4)
    base        = np.random.randn(n_rows, n_latentes)
    mezcla      = np.random.randn(n_latentes, n_sensors)
    X           = base @ mezcla + np.random.randn(n_rows, n_sensors) * 0.3

    y = np.random.randint(0, n_clases, size=n_rows)

    # ------------------------------------------------------------------
    # 3. INPUT
    # ------------------------------------------------------------------
    input_data = {
        'X'               : X.copy(),
        'y'               : y.copy(),
        'varianza_deseada': varianza,
    }

    # ------------------------------------------------------------------
    # 4. OUTPUT esperado  →  replicamos la lógica de reducir_y_clasificar
    # ------------------------------------------------------------------

    # A. Escalar
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # B. PCA con varianza deseada
    pca       = PCA(n_components=varianza, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)
    n_componentes = X_reduced.shape[1]

    # C. Entrenar KNN
    modelo = KNeighborsClassifier()
    modelo.fit(X_reduced, y)

    output_data = (n_componentes, modelo)

    return input_data, output_data


# ----------------------------------------------------------------------
# Ejemplo de uso
# ----------------------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_reducir_y_clasificar()

    print("=== INPUT ===")
    print(f"  Shape de X        : {entrada['X'].shape}")
    print(f"  Shape de y        : {entrada['y'].shape}")
    print(f"  Varianza deseada  : {entrada['varianza_deseada']}")

    print("\n=== OUTPUT ESPERADO ===")
    n_comp, modelo = salida_esperada
    print(f"  Nº componentes PCA : {n_comp}")
    print(f"  Modelo             : {type(modelo).__name__}")
    print(f"  Score (train)      : {modelo.score(
        PCA(n_components=entrada['varianza_deseada'], random_state=42).fit_transform(
            StandardScaler().fit_transform(entrada['X'])
        ), entrada['y']
    ):.4f}")
