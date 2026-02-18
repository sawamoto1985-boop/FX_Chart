import pandas as pd
import sys
import os

# プロジェクトルート（直下のpandas_ta）を検索パスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import pandas_ta as ta
except ImportError:
    print("Error: pandas_ta could not be imported. Please ensure it is in the project root.")

def generate_features(df):
    df = df.sort_values('event_time').copy()
    df.set_index(pd.DatetimeIndex(df['event_time']), inplace=True)
    
    # 以前作成した指標計算（RSI, MACD, BBands等）
    df['sma_20'] = ta.sma(df['close_price'], length=20)
    df['rsi'] = ta.rsi(df['close_price'], length=14)
    
    macd = ta.macd(df['close_price'])
    df = pd.concat([df, macd], axis=1)
    
    bbands = ta.bbands(df['close_price'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    df['returns'] = df['close_price'].pct_change()
    
    # ターゲット作成（5分後）
    df['target'] = (df['close_price'].shift(-5) > df['close_price']).astype(int)
    df.dropna(inplace=True)
    return df
