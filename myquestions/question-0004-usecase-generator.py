import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import random


def generar_caso_de_uso_buscar_mejor_svm():
    """
    Genera un caso de prueba aleatorio (input y output esperado)
    para la función buscar_mejor_svm(X, y).

    La función devuelve:
        input_data  : dict con claves 'X' e 'y'
        output_data : dict con claves 'best_params_' y 'best_score_'
    """

    # ------------------------------------------------------------------
    # 1. Configuración aleatoria
    # ------------------------------------------------------------------
    n_rows     = random.randint(40, 120)   # suficientes filas para CV con k=3
    n_features = random.randint(2, 8)
    n_clases   = random.randint(2, 4)

    # ------------------------------------------------------------------
    # 2. Generar datos de clasificación
    # ------------------------------------------------------------------
    # Datos con cierta estructura para que SVM tenga algo real que aprender
    X = np.random.randn(n_rows, n_features)
    y = np.random.randint(0, n_clases, size=n_rows)

    # ------------------------------------------------------------------
    # 3. INPUT
    # ------------------------------------------------------------------
    input_data = {
        'X': X.copy(),
        'y': y.copy(),
    }

    # ------------------------------------------------------------------
    # 4. OUTPUT esperado  →  replicamos la lógica de buscar_mejor_svm
    # ------------------------------------------------------------------

    # A. Malla de parámetros (igual que debe definir el alumno)
    param_grid = {
        'kernel': ['linear', 'rbf'],
        'C'     : [1, 10],
    }

    # B. GridSearchCV con k=3
    svc        = SVC()
    grid_search = GridSearchCV(svc, param_grid, cv=3)
    grid_search.fit(X, y)

    # C. Resultado esperado
    output_data = {
        'best_params_': grid_search.best_params_,
        'best_score_' : grid_search.best_score_,
    }

    return input_data, output_data


# ----------------------------------------------------------------------
# Ejemplo de uso
# ----------------------------------------------------------------------
if __name__ == "__main__":
    entrada, salida_esperada = generar_caso_de_uso_buscar_mejor_svm()

    print("=== INPUT ===")
    print(f"  Shape de X : {entrada['X'].shape}")
    print(f"  Shape de y : {entrada['y'].shape}")
    print(f"  Clases únicas en y : {sorted(set(entrada['y']))}")

    print("\n=== OUTPUT ESPERADO ===")
    print(f"  Mejores parámetros : {salida_esperada['best_params_']}")
    print(f"  Mejor score (CV=3) : {salida_esperada['best_score_']:.4f}")
