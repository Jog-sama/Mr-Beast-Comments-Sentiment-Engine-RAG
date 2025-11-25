import streamlit as st
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
import os
from collections import Counter

st.set_page_config(page_title="MrBeast Comments Search", layout="wide", page_icon="🎬")

@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path="./chroma_db")

model = load_model()
client = get_client()

if "mrbeast_comments" in [c.name for c in client.list_collections()]:
    coll = client.get_collection("mrbeast_comments")
else:
    coll = client.create_collection("mrbeast_comments")

@st.cache_data
def load_data():
    possible_files = ["sentiment_analysis_dataset.csv", "data.csv"]
    
    df = None
    for filename in possible_files:
        if os.path.exists(filename):
            df = pd.read_csv(filename, on_bad_lines='skip')
            break
    
    if df is None:
        st.error("CSV file not found")
        st.stop()
    
    df = df.fillna("")
    df['Comment'] = df['Comment'].astype(str).str.strip()
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip()
    
    return df

df = load_data()

def build_embeddings():
    with st.spinner("Building embeddings..."):
        try:
            coll.delete(where={})
        except:
            pass
        
        comments = df["Comment"].tolist()
        sentiments = df["Sentiment"].tolist()
        ids = [f"comment_{i}" for i in range(len(comments))]
        metadatas = [{"sentiment": sent} for sent in sentiments]
        
        batch_size = 100
        progress_bar = st.progress(0)
        
        for i in range(0, len(comments), batch_size):
            batch_comments = comments[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            batch_metadata = metadatas[i:i+batch_size]
            batch_embeddings = model.encode(batch_comments, show_progress_bar=False).tolist()
            
            coll.add(documents=batch_comments, ids=batch_ids, embeddings=batch_embeddings, metadatas=batch_metadata)
            
            progress = min(i + batch_size, len(comments))
            progress_bar.progress(progress / len(comments))
        
        progress_bar.empty()

with st.sidebar:
    st.header("📊 Dataset")
    st.metric("Total Comments", f"{len(df):,}")
    
    sentiment_counts = df['Sentiment'].value_counts()
    st.subheader("Sentiment")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / len(df)) * 100
        st.write(f"**{sentiment}**: {count:,} ({percentage:.1f}%)")
    
    st.markdown("---")
    st.subheader("🎯 Filter")
    sentiment_filter = st.multiselect("Sentiment:", options=["All"] + df['Sentiment'].unique().tolist(), default=["All"])
    
    st.markdown("---")
    n_results = st.slider("Results", 5, 50, 10)

st.title("🎬 MrBeast Comments Search")

col1, col2 = st.columns([3, 1])

with col2:
    if st.button("🔄 Build Embeddings", use_container_width=True):
        build_embeddings()
        st.success("✅ Done!")
        st.balloons()

with col1:
    query = st.text_input("🔍 Search", placeholder="funny reactions, criticism, money...")

if query:
    try:
        q_emb = model.encode([query]).tolist()
        
        where_filter = None
        if "All" not in sentiment_filter and len(sentiment_filter) > 0:
            where_filter = {"sentiment": sentiment_filter[0]} if len(sentiment_filter) == 1 else {"sentiment": {"$in": sentiment_filter}}
        
        res = coll.query(query_embeddings=q_emb, n_results=n_results, where=where_filter)
        
        if res["documents"][0]:
            st.markdown(f"### 🎯 Top {len(res['documents'][0])} Results")
            
            result_sentiments = [meta["sentiment"] for meta in res["metadatas"][0]]
            sentiment_dist = Counter(result_sentiments)
            
            cols = st.columns(len(sentiment_dist))
            for idx, (sent, count) in enumerate(sentiment_dist.items()):
                with cols[idx]:
                    emoji = "😊" if sent == "Positive" else "😐" if sent == "Neutral" else "😞"
                    st.metric(f"{emoji} {sent}", count)
            
            st.markdown("---")
            
            docs = res["documents"][0]
            metadatas = res["metadatas"][0]
            
            for idx, (comment, metadata) in enumerate(zip(docs, metadatas), 1):
                sentiment = metadata["sentiment"]
                
                if sentiment == "Positive":
                    emoji, color = "😊", "#90EE90"
                elif sentiment == "Negative":
                    emoji, color = "😞", "#FFB6C6"
                else:
                    emoji, color = "😐", "#FFE4B5"
                
                col_a, col_b = st.columns([0.9, 0.1])
                with col_a:
                    st.markdown(f"**Result {idx}**")
                with col_b:
                    st.markdown(f"<div style='background-color: {color}; padding: 5px 10px; border-radius: 15px; text-align: center;'>{emoji} {sentiment}</div>", unsafe_allow_html=True)
                
                st.write(f"_{comment}_")
                st.markdown("---")
        else:
            st.warning("No results. Build embeddings first.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("💡 Build embeddings first.")

with st.expander("💡 Examples"):
    st.markdown("- `funny reactions`\n- `criticism`\n- `money comments`\n- `ethical concerns`")

st.caption(f"🎮 {len(df):,} comments")