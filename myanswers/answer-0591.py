import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

def log_verosimilitud_promedio_gmm(df):
    # 1. Estandarizamos los datos que recibe la función
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # 2. Creamos y ajustamos el modelo con los parámetros exactos que pide el compañero
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(df_scaled)
    
    # 3. Calculamos la log-verosimilitud y la devolvemos garantizando que sea float
    return float(gmm.score(df_scaled))
