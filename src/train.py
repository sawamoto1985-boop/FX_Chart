import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from features import generate_features
from supabase import create_client, Client

# Supabase設定
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def load_data_from_supabase():
    # 全データを取得（データ量が多い場合は取得期間を制限してください）
    response = supabase.table("fx_candles_1m").select("*").execute()
    df = pd.DataFrame(response.data)
    return df

def train():
    # 1. データの読み込みと特徴量生成
    raw_df = load_data_from_supabase()
    df = generate_features(raw_df)

    # 2. 特徴量(X)と正解ラベル(y)に分割
    # features.pyで作成されたカラムを特定（不要なカラムを除外）
    drop_cols = ['id', 'pair_name', 'event_time', 'target']
    X = df.drop(columns=drop_cols)
    y = df['target']

    # 3. 訓練データとテストデータに分割（時系列を維持するため shuffle=False）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # 4. LightGBMモデルの設定
    # 精度重視のパラメータ設定
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': 0.01, # 低学習率でじっくり学習
        'num_leaves': 64,      # 複雑なパターンを捉える
        'feature_fraction': 0.8
    }

    # 5. 学習実行
    print("AIモデルの学習を開始します...")
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )

    # 6. 精度評価
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # 7. モデルの保存
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/fx_ai_model.pkl')
    print("モデルを models/fx_ai_model.pkl に保存しました。")

if __name__ == "__main__":
    train()
