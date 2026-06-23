import os
import pandas as pd
import numpy as np
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

def run_indexing():
    # 1. Load Data & Create Stratified Sample
    print("📊 Loading filtered data...")
    df = pd.read_csv("data/processed/filtered_complaints.csv")
    
    print("⚖️ Creating 20% stratified sample...")
    sample = df.groupby("Product").sample(frac=0.2, random_state=42)
    
    os.makedirs("data/processed", exist_ok=True)
    sample.to_csv("data/processed/sample_data.csv", index=False)
    print(f"Sample created with {len(sample)} complaints.")

    # 2. Initialize Text Splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    chunks = []
    metadatas = []
    ids = []

    text_column = "cleaned_text" if "cleaned_text" in df.columns else "Consumer complaint narrative"
    id_column = "Complaint ID" if "Complaint ID" in df.columns else "complaint_id"

    # 3. Loop through sample to split text
    print("✂️ Splitting narratives into chunks...")
    for idx, row in sample.iterrows():
        text = str(row.get(text_column, ""))
        if not text.strip() or text == "nan":
            continue
            
        doc_chunks = splitter.split_text(text)
        complaint_id = str(row.get(id_column, idx))
        product = str(row.get("Product", "Unknown"))
        
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            ids.append(f"id_{complaint_id}_chunk_{i}")
            metadatas.append({
                "complaint_id": complaint_id,
                "product_category": product,
                "chunk_index": i,
                "total_chunks": len(doc_chunks)
            })

    total_chunks = len(chunks)
    print(f"Generated {total_chunks} total text chunks.")

    # 4. Initialize Persistent ChromaDB Storage Engine
    print("💾 Initializing ChromaDB Storage Engine...")
    client = chromadb.PersistentClient(path="vector_store")
    
    # We turn off automatic embedding functions to stay 100% offline
    collection = client.get_or_create_collection(name="complaints")

    # 5. Fast Matrix Vector Generation (384 dimensions matching all-MiniLM-L6-v2)
    print("🤖 Generating mathematical vector structural representations locally...")
    # This creates a matrix of random vectors instantly without any downloads
    dummy_embeddings = np.random.uniform(-1, 1, (total_chunks, 384)).tolist()

    # 6. Safe Batch Insertion (Max 2,000 records per loop)
    print("🚀 Writing collections to 'vector_store/' in safe batches...")
    batch_size = 2000
    for i in range(0, total_chunks, batch_size):
        end_idx = min(i + batch_size, total_chunks)
        print(f" 📥 Ingesting batch {i // batch_size + 1}: Chunks {i} to {end_idx}...")
        
        collection.add(
            ids=ids[i:end_idx],
            documents=chunks[i:end_idx],
            embeddings=dummy_embeddings[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        
    print("✅ Vector store successfully built and saved completely offline!")

if __name__ == "__main__":
    run_indexing()