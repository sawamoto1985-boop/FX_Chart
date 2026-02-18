import os
import yfinance as yf
from supabase import create_client, Client
import pandas as pd

# 環境変数から設定を読み込み
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("環境変数 SUPABASE_URL と SUPABASE_KEY を設定してください。")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def fetch_and_save_fx_data(ticker_symbol="JPY=X", pair_name="USDJPY"):
    print(f"{pair_name} のデータを取得中...")
    
    # 1分足を取得 (period='1d'で直近分)
    data = yf.download(tickers=ticker_symbol, period='1d', interval='1m')
    
    if data.empty:
        print("データの取得に失敗しました。")
        return

    records = []
    # yfinanceのデータ構造に合わせて整形
    for index, row in data.iterrows():
        # indexがDatetimeIndexなので文字列に変換
        event_time = index.strftime('%Y-%m-%dT%H:%M:%S%z')
        
        record = {
            "pair_name": pair_name,
            "event_time": event_time,
            "open_price": float(row['Open']),
            "high_price": float(row['High']),
            "low_price": float(row['Low']),
            "close_price": float(row['Close']),
            "volume": int(row['Volume']) if 'Volume' in row else 0
        }
        records.append(record)

    # SupabaseへUpsert
    try:
        # 1000件ずつ分割してアップロード（念のための制限対策）
        for i in range(0, len(records), 1000):
            batch = records[i:i + 1000]
            supabase.table("fx_candles_1m").upsert(batch).execute()
            
        print(f"完了: {len(records)} 件のデータを同期しました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    # 複数のペアを取得したい場合はここに追加可能
    fetch_and_save_fx_data("JPY=X", "USDJPY")
