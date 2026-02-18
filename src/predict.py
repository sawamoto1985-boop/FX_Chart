import os
import joblib
import pandas as pd
from datetime import datetime
from supabase import create_client, Client
from features import generate_features

# 設定の読み込み
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def predict_minute_by_minute():
    # 1. 最新の1分足を少し多めに取得（計算用に直近100件）
    res = supabase.table("fx_candles_1m") \
        .select("*") \
        .order("event_time", desc=True) \
        .limit(100) \
        .execute()
    
    df_raw = pd.DataFrame(res.data)
    if len(df_raw) < 50:
        print("データが不足しています。")
        return

    # 2. 特徴量を計算（自作のfeatures.pyを使用）
    df_features = generate_features(df_raw)
    
    # 3. 直近5分間のデータを抽出
    # GitHub Actionsが5分おきに動くので、その間の「1分ごとの隙間」を埋める
    recent_data = df_features.tail(5)
    
    # 4. モデル読み込み
    model_path = 'models/fx_ai_model.pkl'
    if not os.path.exists(model_path):
        print("モデルが見つかりません。")
        return
    model = joblib.load(model_path)

    # 5. 1分ごとにループして予測
    for _, row in recent_data.iterrows():
        # 予測に必要な特徴量だけに絞る
        X = pd.DataFrame([row]).drop(columns=['id', 'pair_name', 'event_time', 'target'], errors='ignore')
        
        prediction = int(model.predict(X)[0])
        prob = model.predict_proba(X)[0]
        
        # 保存用データ作成
        result = {
            "pair_name": "USDJPY",
            "target_time": row['event_time'], # その1分ごとの時間
            "direction": "UP" if prediction == 1 else "DOWN",
            "confidence": float(max(prob) * 100)
        }
        
        # upsert（重複があれば更新、なければ挿入）で保存
        # これにより「1分単位」のシグナルがDBに蓄積される
        supabase.table("ai_predictions").upsert(
            result, on_conflict="pair_name,target_time"
        ).execute()

    print(f"完了: 直近{len(recent_data)}分間の予測を更新しました。")

if __name__ == "__main__":
    predict_minute_by_minute()
