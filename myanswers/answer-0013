def preparar_datos(df, target_col):
    """
    Separa las características de la etiqueta, imputa valores faltantes 
    con la media y escala los datos numéricos.
    """
    # 1. Separar características (X) y la etiqueta (y)
    X_raw = df.drop(columns=[target_col])
    y = df[target_col].values
    
    # 2. Imputar los valores nulos (NaN) usando la media
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_raw)
    
    # 3. Escalar las características para tener media 0 y desviación estándar 1
    scaler = StandardScaler()
    X_procesado = scaler.fit_transform(X_imputed)
    
    # 4. Retornar la tupla esperada
    return X_procesado, y
