import os
import pandas as pd
import lightgbm as lgb
import joblib
from sklearn.model_selection import train_test_split
from supabase import create_client, Client
from features import generate_features  # 先ほど作った自前計算版のfeatures

# 1. 設定の読み込み
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def train_model():
    print("1. Supabaseから学習用データを取得中...")
    # 直近5000件程度のデータを取得して学習に使用
    response = supabase.table("fx_candles_1m").select("*").order("event_time", desc=True).limit(5000).execute()
    raw_df = pd.DataFrame(response.data)
    
    if len(raw_df) < 100:
        print("データが少なすぎるため、学習をスキップします。")
        return

    print("2. 特徴量を生成中...")
    # pandas_taを使わない自作のfeatures.pyを使用
    df = generate_features(raw_df)

    # 3. 学習用データ(X)と正解(y)に分ける
    # 予測に使わないカラムを除去
    drop_cols = ['id', 'pair_name', 'event_time', 'target']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['target']

    # 4. データを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("3. AIの学習（頭脳の作成）を開始...")
    # 精度と速度のバランスが良い設定
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='binary_logloss',
        learning_rate=0.05,
        n_estimators=100,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train, y_train)

    # 5. モデルを保存（GitHub Actions内で次ステップに渡すため）
    print("4. 学習済みモデルを保存中...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fx_ai_model.pkl')
    
    print("--- 学習完了: models/fx_ai_model.pkl を作成しました ---")

if __name__ == "__main__":
    train_model()
