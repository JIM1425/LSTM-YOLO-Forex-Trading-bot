import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time
from datetime import datetime, timedelta
import os
import mplfinance as mpf
from ultralytics import YOLO
from PIL import Image


# conectar con MT5
mt5.initialize()

symbol_yolo = "EURUSD"           
minutos_yolo = 30    
espera_segundos = 5 * 60  # Esperar x minutos si no hay patrón           


while True:
    
    # Descargar datos desde MT5
    cantidad_barras = minutos_yolo
    mt5.symbol_select(symbol_yolo, True)
    rates_yolo = mt5.copy_rates_from_pos(symbol_yolo, mt5.TIMEFRAME_M1, 0, cantidad_barras)
    if rates_yolo is None or len(rates_yolo) == 0:
        print("No se pudieron obtener datos para el gráfico de velas.")
    else:
        df_yolo = pd.DataFrame(rates_yolo)
        df_yolo['time'] = pd.to_datetime(df_yolo['time'], unit='s')
        df_yolo.set_index('time', inplace=True)

        # Graficar velas japonesas
        os.makedirs("temp", exist_ok=True)
        img_path = "temp/ultima_vela.png"

        mpf.plot(df_yolo[['open', 'high', 'low', 'close']], type='candle', style='charles',
                savefig=img_path, figsize=(8, 4))
        
        # Deteccion con modelo
        modelo_yolo = YOLO("best.pt")  
        resultados = modelo_yolo('img1.png')
        etiquetas = [modelo_yolo.names[int(box.cls)] for r in resultados for box in r.boxes]
    
        
    
    if not etiquetas:
        print("YOLO no detectó ningún patrón. Esperando 5 minutos para reintentar...")
        time.sleep(espera_segundos)
        continue

    patron_detectado = etiquetas[0].lower()
    if patron_detectado in ['head and shoulders', 'double top']:
        tendencia_yolo = 'baja'
    elif patron_detectado in ['double bottom', 'rounding bottom', 'cup and handle']:
        tendencia_yolo = 'sube'
    else:
        tendencia_yolo = 'desconocida'

    print(f"YOLO detectó patrón: {patron_detectado} → Tendencia: {tendencia_yolo}")
    
    tendencia_yolo 
    if tendencia_yolo in ['sube', 'baja']:
        print("Activando predicción LSTM...")

        

    
    # ----- Modelo LSTM  ----- #

    symbol = "EURUSD"
    end = datetime.now()
    start = end - timedelta(days=2)

    # Descargar velas de 1 minuto
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)

    # Convertir a DataFrame 
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Preparar datos, normalizar 
    data = df[['open', 'high', 'low', 'close', 'tick_volume']].values
    target = df['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data)
    target_scaled = scaler.fit_transform(target.reshape(-1, 1))

    # Crear ventanas 
    def crear_ventanas(series, target, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(target[i+window_size])
        return np.array(X), np.array(y)

    window_size = 60  # Tamaño de ventana en minutos
    X, y = crear_ventanas(data_scaled, target_scaled, window_size)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window_size, 5)),
        Dropout(0.2),  

        LSTM(32),
        Dropout(0.2),

        Dense(16, activation='relu'),
        Dense(1, activation='linear')  # Para regresión: predicción continua
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Entrenar
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)

    # Evaluar
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print('test_mae:', test_mae)
    print('test_loss_', test_loss)


    # Predecir los próximos minutos
    minutos = 5
    last_seq = data_scaled[-window_size:]
    predictions = []

    for _ in range(minutos):
        x_input = last_seq.reshape(1, window_size, 5)
        pred_scaled = model.predict(x_input, verbose=0)
        predictions.append(pred_scaled[0, 0])
        # Generar una nueva fila con 5 features falsas para mantener dimensión
        new_row = pred_scaled.reshape(1, -1).repeat(5, axis=1)  # igualar a 5 columnas
        last_seq = np.append(last_seq[1:], new_row, axis=0)

    # Desescalar predicciones
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    # Generar timestamps futuros
    last_times = pd.date_range(df['time'].iloc[-1], periods=minutos + 1, freq='1min')[1:]

    # Graficar
    plt.figure(figsize=(12, 4))
    plt.plot(df['time'], df['close'], label="Precio real")
    plt.plot(last_times, predicted_prices, label="Predicción LSTM", color="red", marker='o')
    plt.title(f"Predicción de los próximos {minutos} minutos")
    plt.xlabel("Tiempo")
    plt.ylabel("Precio de cierre")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ----- TENDENCIA ------ #
    last_real = df['close'].iloc[-1]
    last_pred = predicted_prices[-1][0]
    tendencia_lstm = "sube" if last_pred > last_real else "baja"

    print(f"Tendencia predicha: El precio {tendencia_lstm}")
    print(f"Último precio real: {last_real:.5f}")
    print(f"Último precio predicho: {last_pred:.5f}")


    # ------ DECISION -------- #

    if tendencia_lstm == tendencia_yolo:
        print(f"Tendencia confirmada por ambos modelos: {tendencia_lstm.upper()}")
        tipo = "BUY" if tendencia_lstm == "sube" else "SELL"


        # Definir TP/SL (Take Profit, Stop Loss)

        margen_factor = 50    # para ajustal el tp/sl

        pred_change = last_pred - last_real
        pips = pred_change / 0.0001  # 1 pip = 0.0001 en EURUSD

        # Aplica el margen
        if pips > 0:
            take_profit = last_real + abs(pred_change * margen_factor)
            stop_loss = last_real - abs(pred_change * 0.5 * margen_factor)
            tipo = "BUY"
        else:
            take_profit = last_real - abs(pred_change * margen_factor)
            stop_loss = last_real + abs(pred_change * 0.5 * margen_factor)
            tipo = "SELL"

        print(f"{tipo} con TP: {take_profit:.5f}, SL: {stop_loss:.5f}, Cambio estimado: {pips:.1f} pips")

        # ------ GENERAR ORDEN ------ #

        symbol = "EURUSD"
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("No se pudo obtener el precio del símbolo.")
            mt5.shutdown()
            quit()

        price = tick.ask if tipo == "BUY" else tick.bid

        order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": 0.1,
            "type": mt5.ORDER_TYPE_BUY if tipo == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Bot LSTM+YOLO",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
            "sl": stop_loss,
            "tp": take_profit,
        }

        result = mt5.order_send(order)
        print("Resultado:", result)

        # ------ CIERRE DE OPERACION ------ # 

        print(f"Esperando {minutos} minutos antes de cerrar la operación...")
        time.sleep(minutos * 60)

        ticket = result.order  # ID de la orden abierta
        positions = mt5.positions_get(ticket=ticket)

        if positions:
            pos = positions[0]
            cierre = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": pos.ticket,
                "price": mt5.symbol_info_tick(pos.symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(pos.symbol).ask,
                "deviation": 10,
                "magic": 123456,
                "comment": "Cierre automático por tiempo",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            cierre_resultado = mt5.order_send(cierre)
            print("Resultado del cierre:", cierre_resultado)
        else:
            print("La operación ya fue cerrada antes de tiempo.")

        time.sleep(espera_segundos)
        print(f'Esperando {espera_segundos} minutos antes de volver a intentar')
        continue

    else:
        print("Tendencias no coinciden (YOLO: {}, LSTM: {}). No se opera.".format(tendencia_yolo, tendencia_lstm))
        print(f"Esperando {espera_segundos // 60} minutos antes de volver a intentar...")
        time.sleep(espera_segundos)
        continue 
        

mt5.shutdown()


