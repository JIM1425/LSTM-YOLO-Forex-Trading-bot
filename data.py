import numpy as np
import pandas as pd
import mplfinance as mpf
import os

# Crear carpetas para guardar
patrones = ["head_and_shoulders", "double_top", "double_bottom", "rounding_bottom", "cup_and_handle"]
for p in patrones:
    os.makedirs(f"dataset/{p}", exist_ok=True)

def generar_patron(patron, num=10):
    for i in range(num):
        df = crear_datos_patron(patron)
        if df is not None:
            path = f"dataset/{patron}/{patron}_{i}.png"
            mpf.plot(df, type='candle', style='charles', savefig=path)

# Funciones generadoras sintéticas
def crear_datos_patron(patron):
    np.random.seed()  # diferente cada vez
    base = np.linspace(1, 1.2, 60)

    if patron == "double_top":
        shape = np.concatenate([
            np.linspace(1.0, 1.15, 15),
            np.linspace(1.15, 1.05, 5),
            np.linspace(1.05, 1.15, 10),
            np.linspace(1.15, 1.00, 15),
        ])
    elif patron == "double_bottom":
        shape = np.concatenate([
            np.linspace(1.2, 1.05, 15),
            np.linspace(1.05, 1.15, 5),
            np.linspace(1.15, 1.05, 10),
            np.linspace(1.05, 1.20, 15),
    
        ])
    elif patron == "head_and_shoulders":
        shape = np.concatenate([
            np.linspace(1.0, 1.1, 10),
            np.linspace(1.1, 1.05, 5),
            np.linspace(1.05, 1.2, 10),
            np.linspace(1.2, 1.05, 10),
            np.linspace(1.05, 1.1, 10),
            np.linspace(1.1, 1.0, 15)
        ])
    elif patron == "rounding_bottom":
        shape = 1.05 - 0.05 * np.cos(np.linspace(np.pi, 3*np.pi, 60))
    elif patron == "cup_and_handle":
        curva = 1.05 - 0.05 * np.cos(np.linspace(np.pi, 3*np.pi, 40))
        handle = np.linspace(curva[-1], curva[-1] - 0.02, 10)
        shape = np.concatenate([curva, handle])
    else:
        return None

    noise = np.random.normal(0, 0.001, size=shape.shape)
    shape += noise

    df = pd.DataFrame({
        'Open': shape + np.random.normal(0, 0.002, size=len(shape)),
        'High': shape + np.random.normal(0.005, 0.001, size=len(shape)),
        'Low':  shape - np.random.normal(0.005, 0.001, size=len(shape)),
        'Close': shape + np.random.normal(0, 0.002, size=len(shape)),
    })

    df['Date'] = pd.date_range("2025-01-01", periods=len(df), freq='min')
    df.set_index('Date', inplace=True)
    return df

# Ejecutar para todos los patrones
for patron in patrones:
    generar_patron(patron, num=10)