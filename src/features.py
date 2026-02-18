import pandas as pd
import numpy as np

def generate_features(df):
    # データを時間順に並び替え
    df = df.sort_values('event_time').copy()
    
    # 1. シンプルな移動平均 (SMA)
    # pandas_ta.sma(close, length=20) と同じ計算です
    df['sma_20'] = df['close_price'].rolling(window=20).mean()
    
    # 2. RSI (相対力指数) の自作計算
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # 3. 変化率 (Returns)
    df['returns'] = df['close_price'].pct_change()
    
    # 4. ラグ特徴量（直近3分間の動き）
    for i in range(1, 4):
        df[f'lag_returns_{i}'] = df['returns'].shift(i)

    # 5. 正解ラベル（5分後の価格が上がったか）
    df['target'] = (df['close_price'].shift(-5) > df['close_price']).astype(int)

    # 計算できない最初の数行を削除
    df.dropna(inplace=True)
    
    return df
