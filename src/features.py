import pandas as pd
import pandas_ta as ta

def generate_features(df):
    """
    生の1分足データからAI学習用の特徴量を生成する
    df: columns=['event_time', 'open_price', 'high_price', 'low_price', 'close_price', 'volume']
    """
    # データを時間順に並び替え、型を整える
    df = df.sort_values('event_time').copy()
    
    # pandas_taを使うための準備（カラム名を合わせる）
    df.set_index(pd.DatetimeIndex(df['event_time']), inplace=True)
    ohlcv = df[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
    ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']

    # --- テクニカル指標の追加 ---
    # 1. トレンド系 (Moving Averages)
    df['sma_20'] = ta.sma(ohlcv['close'], length=20)
    df['ema_50'] = ta.ema(ohlcv['close'], length=50)
    
    # 2. オシレーター系 (Momentum)
    df['rsi'] = ta.rsi(ohlcv['close'], length=14)
    macd = ta.macd(ohlcv['close'])
    df = pd.concat([df, macd], axis=1) # MACD_12_26_9 等が追加される
    
    # 3. ボラティリティ系 (Volatility)
    bbands = ta.bbands(ohlcv['close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    df['atr'] = ta.atr(ohlcv['high'], ohlcv['low'], ohlcv['close'], length=14)
    
    # 4. 統計的特徴 (Returns & Lags)
    # 前の足との変化率（これがAIにとって重要）
    df['returns'] = ohlcv['close'].pct_change()
    
    # ラグ特徴量：直近3分間の動きを個別の列にする
    for i in range(1, 4):
        df[f'lag_rsi_{i}'] = df['rsi'].shift(i)
        df[f'lag_returns_{i}'] = df['returns'].shift(i)

    # --- 正解ラベル（Target）の作成 ---
    # 今回は「5分後の価格が今の価格より高いか(1)低いか(0)」を予測対象にする
    # 精度を出すために、少し先の未来を予測させます
    df['target'] = (ohlcv['close'].shift(-5) > ohlcv['close']).astype(int)

    # 欠損値（計算に必要なデータが足りない行）を削除
    df.dropna(inplace=True)
    
    return df

if __name__ == "__main__":
    # テスト用のダミー実行（実際にはSupabaseから取得したdfを渡す）
    print("Feature generation logic ready.")
