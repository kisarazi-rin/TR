import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==== Webアプリのタイトル ====
st.title("機械学習")
st.write("CSVデータを読み込んでAIを学習させ、条件を入力して温度を予測")

# ==== 1. データ読み込み (ファイルアップロード機能) ====
uploaded_file = st.file_uploader("学習用CSVファイル (engine_data.csv) をアップロード", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("データのプレビュー:")
    st.dataframe(df.head()) # データの中身を少し表示

    # 使う特徴量の列名
    # ※ CSVにこの列名が含まれている必要があります
    feature_cols = ["timing_deg", "rpm"]
    target_col = "wall_temp_max"

    # カラムの存在チェック
    if all(col in df.columns for col in feature_cols + [target_col]):
        
        X = df[feature_cols]
        y = df[target_col]

        # ==== 2. 学習用 / テスト用 に分割 ====
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        # ==== 3. モデル学習 (ボタンを押したら学習) ====
        # 毎回学習し直すと重いので、キャッシュ機能を使っても良いですが今回はシンプルにします
        model = RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)
        model.fit(X_train, y_train)

        # ==== 4. 性能評価 ====
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("モデルの学習が完了しました！")
        col1, col2 = st.columns(2)
        col1.metric("MAE (平均絶対誤差)", round(mae, 2))
        col2.metric("R^2 (決定係数)", round(r2, 4))

        st.markdown("---")

        # ==== 5. GUIで新しい条件を入力して予測 ====
        st.subheader("予測シミュレーション")

        # サイドバーまたはメイン画面に入力フォームを作る
        input_timing = st.number_input("点火時期 (timing_deg)", value=-3.0, step=0.1)
        input_rpm = st.number_input("回転数 (rpm)", value=3000, step=100)

        # 予測ボタン
        if st.button("温度を予測する"):
            # 入力データを作る
            new_cond = pd.DataFrame([{
                "timing_deg": input_timing,
                "rpm": input_rpm,
            }])
            
            # 予測実行
            pred_temp = model.predict(new_cond)[0]
            
            # 結果表示
            st.write(f"### 予想壁温度: {pred_temp:.2f} ℃")

    else:
        st.error(f"CSVファイルに {feature_cols} または {target_col} の列が見つかりません。")

else:
    st.info("CSVファイルをアップロードすると学習が始まります。")
