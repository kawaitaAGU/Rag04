import streamlit as st
import pandas as pd
import pickle
from openai import OpenAI
from pathlib import Path
import time

# ✅ secrets.toml から API キー取得
if "OPENAI_API_KEY" not in st.secrets:
    st.error("OPENAI_API_KEY が設定されていません。")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ CSVファイル読み込み
csv_path = Path("sample.csv")
if not csv_path.exists():
    st.error(f"CSVファイルが見つかりませんでした: {csv_path}")
    st.stop()

df = pd.read_csv(csv_path)

# ✅ 設問列からテキスト抽出
texts = df["設問"].astype(str).tolist()
embeddings = []

st.write(f"✅ 全 {len(texts)} 問に対してEmbeddingを作成します。")

progress_bar = st.progress(0)

# ✅ 埋め込み生成（進捗表示付き）
for i, text in enumerate(texts):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        embeddings.append(response.data[0].embedding)
    except Exception as e:
        st.error(f"エラー（{i}番目）: {e}")
        embeddings.append([0.0] * 1536)  # エラー時はダミーで埋める

    progress_bar.progress((i + 1) / len(texts))

# ✅ Macのデスクトップに保存
desktop_path = Path.home() / "Desktop" / "embeddings.pkl"
with open(desktop_path, "wb") as f:
    pickle.dump({"embeddings": embeddings, "df": df}, f)

st.success(f"✅ デスクトップに保存しました: {desktop_path}")
