import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import accuracy_score

def propagar_etiquetas_label_spreading(df, target_col, features=None, kernel="rbf", gamma=20, unlabeled_fraction=0.6, test_size=0.2, random_state=42):
    """
    Entrena un modelo semi-supervisado LabelSpreading propagando etiquetas 
    tras imputar valores nulos y escalar los datos.
    """
    # 1. Seleccionar características (si es None, usa todas las numéricas menos target)
    if features is None:
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in features:
            features.remove(target_col)
            
    # 2. Filtrar solo las filas que tienen una etiqueta conocida (que no son NaN en target)
    df_known = df[df[target_col].notna()].copy()
    X_known = df_known[features].to_numpy()
    y_known = df_known[target_col].astype(int).to_numpy()
    
    # 3. Separar en Train y Test de forma estratificada
    X_train, X_test, y_train_full, y_test = train_test_split(
        X_known, 
        y_known, 
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y_known
    )
    
    # 4. Ocultar aleatoriamente una fracción de las etiquetas en Train (asignarles -1)
    rng2 = np.random.default_rng(int(random_state))
    n_train = len(y_train_full)
    n_unlab = int(np.clip(round(unlabeled_fraction * n_train), 0, n_train - 1))
    
    mask_unlab = np.zeros(n_train, dtype=bool)
    if n_unlab > 0:
        mask_unlab[rng2.choice(n_train, size=n_unlab, replace=False)] = True
        
    y_train_semi = y_train_full.copy()
    y_train_semi[mask_unlab] = -1
    
    # 5. Imputar nulos, Escalar y ajustar LabelSpreading
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    ls = LabelSpreading(kernel=kernel, gamma=float(gamma))
    
    # Preprocesar Train y entrenar
    X_train_imp = imputer.fit_transform(X_train)
    X_train_sca = scaler.fit_transform(X_train_imp)
    ls.fit(X_train_sca, y_train_semi)
    
    # Preprocesar Test y predecir
    X_test_imp = imputer.transform(X_test)
    X_test_sca = scaler.transform(X_test_imp)
    y_pred = ls.predict(X_test_sca).astype(int)
    
    # 6. Calcular Accuracy
    acc = float(accuracy_score(y_test, y_pred))
    
    # 7. Empaquetar todo en un Pipeline para devolverlo
    pipe = Pipeline(steps=[
        ("imputer", imputer),
        ("scaler", scaler),
        ("ls", ls)
    ])
    
    # En este caso devolvemos una tupla (en lugar de un diccionario) como pide el enunciado
    return y_test.astype(int), y_pred, acc, pipe
