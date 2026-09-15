import os
import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

PAIRS_CSV_PATH = "Dataset/amazon_pairs.csv"
GOLDEN_SET_PATH = "Dataset/golden_dataset.csv"
CHROMA_PATH = "./chroma_db"

def build_vector_db(sample_size: int = 3000):
    df = pd.read_csv(PAIRS_CSV_PATH).dropna()
    golden_df = pd.read_csv(GOLDEN_SET_PATH).dropna(subset=['clean_customer'])
    
    golden_queries = set(golden_df['clean_customer'].astype(str).str.strip().str.lower())
    train_df = df[~df['clean_customer'].astype(str).str.strip().str.lower().isin(golden_queries)]
    
    print(f"Total dataset pairs: {len(df)}")
    print(f"Filtered out {len(df) - len(train_df)} golden evaluation set queries to prevent data leakage.")
    
    sampled_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)

    documents = []
    for _, row in sampled_df.iterrows():
        doc = Document(
            page_content=f"Customer Issue: {row['clean_customer']}",
            metadata={"historical_reply": row['clean_response']}
        )
        documents.append(doc)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print(f"Successfully indexed {len(documents)} leakage-free pairs into ChromaDB!")

if __name__ == "__main__":
    build_vector_db(sample_size=3000)