import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score

def clasificacion_multietiqueta(X_train, y_train, X_test, y_test):
    """
    Entrena un modelo MultiOutputClassifier usando RandomForest
    y calcula el f1-score por etiqueta y su promedio.
    """
    # 1. Instanciar el modelo con los parámetros requeridos
    modelo = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=42)
    )
    
    # 2. Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    
    # 3. Generar predicciones sobre los datos de prueba
    y_pred = modelo.predict(X_test)
    
    # 4. Calcular el F1-score por etiqueta
    f1_etiquetas = f1_score(y_test, y_pred, average=None, zero_division=0)
    
    # 5. Devolver el diccionario con el formato exacto esperado
    resultado = {
        'f1_por_etiqueta': [float(v) for v in f1_etiquetas],
        'f1_promedio': float(np.mean(f1_etiquetas)),
        'n_etiquetas': int(y_test.shape[1])  # Se extrae el número de columnas (etiquetas)
    }
    
    return resultado
