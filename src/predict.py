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

def predict_latest():
    # 1. 最新の1分足をSupabaseから取得（特徴量計算のために直近100件程度）
    response = supabase.table("fx_candles_1m") \
        .select("*") \
        .order("event_time", desc=True) \
        .limit(100) \
        .execute()
    
    df_raw = pd.DataFrame(response.data)
    
    # 2. 特徴量の生成
    df_features = generate_features(df_raw)
    latest_row = df_features.tail(1) # 最新の1行だけを使用
    
    if latest_row.empty:
        print("予測に必要なデータが不足しています。")
        return

    # 3. 学習済みモデルの読み込み
    model_path = 'models/fx_ai_model.pkl'
    if not os.path.exists(model_path):
        print("モデルファイルが見つかりません。train.pyを先に実行してください。")
        return
    model = joblib.load(model_path)

    # 4. 予測の実行
    # 特徴量から予測対象外のカラムを除去
    X = latest_row.drop(columns=['id', 'pair_name', 'event_time', 'target'])
    
    # 予測（0: 下落, 1: 上昇）と、その確率（確信度）を取得
    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities) * 100)
    
    direction = "UP" if prediction == 1 else "DOWN"

    # 5. 結果を ai_predictions テーブルに保存
    prediction_data = {
        "pair_name": "USDJPY",
        "target_time": datetime.now().isoformat(),
        "direction": direction,
        "confidence": confidence
    }

    try:
        supabase.table("ai_predictions").insert(prediction_data).execute()
        print(f"予測完了: {direction} ({confidence:.2f}%) を保存しました。")
    except Exception as e:
        print(f"予測結果の保存に失敗しました: {e}")

if __name__ == "__main__":
    predict_latest()
