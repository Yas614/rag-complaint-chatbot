# src/rag_engine.py
import os
import chromadb
import requests
import re

class CrediTrustRAG:
    def __init__(self, vector_store_path="vector_store"):
        self.client = chromadb.PersistentClient(path=vector_store_path)
        self.collection = self.client.get_collection(name="complaints")
        self.api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        self.headers = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            self.headers["Authorization"] = f"Bearer {hf_token}"

    def retrieve_context(self, query, product_filter=None, k=4):
        where_clause = {}
        if product_filter and product_filter != "All":
            where_clause = {"product_category": product_filter}
            
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where_clause if where_clause else None
        )
        
        retrieved_docs = results.get("documents", [[]])[0]
        retrieved_metadata = results.get("metadatas", [[]])[0]
        return retrieved_docs, retrieved_metadata

    def generate_local_fallback(self, context_chunks):
        """Generates a localized summary directly from the chunks when offline."""
        all_text = " ".join(context_chunks).lower()
        # Simple extraction of common complaint themes for fallback
        keywords = {
            "fees / charges": ["fee", "charge", "interest", "annual", "billed", "overdraft"],
            "account locks / access": ["lock", "block", "access", "freeze", "login", "password"],
            "reporting issues": ["report", "credit score", "bureau", "late payment", "history"],
            "transaction processing": ["transfer", "pending", "transaction", "delay", "sent", "received"]
        }
        
        found_themes = []
        for theme, words in keywords.items():
            if any(w in all_text for w in words):
                found_themes.append(theme)
                
        theme_str = ", ".join(found_themes) if found_themes else "general service irregularities"
        
        return (
            "⚠️ [OFFLINE FALLBACK MODE ACTIVE]\n\n"
            f"Based on an automated linguistic evaluation of the retrieved local records, "
            f"the primary issues detected center around: **{theme_str}**.\n\n"
            "Please review the explicit transaction transcripts attached below for specific details."
        )

    def generate_answer(self, query, context_chunks):
        context_str = "\n\n".join([f"--- Excerpt ---\n{chunk}" for chunk in context_chunks])
        
        prompt = f"""<|system|>
You are an expert financial analytics assistant for CrediTrust Financial. 
Your objective is to provide executive, data-backed insights on customer complaints based ONLY on the context excerpts provided below.

Context Excerpts:
{context_str}
</s>
<|user|>
Question: {query}
</s>
<|assistant|>
Answer:"""

        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 400, "temperature": 0.2, "top_p": 0.9}
        }
        
        try:
            # Short timeout to fail quickly if internet drops
            response = requests.post(self.api_url, json=payload, headers=self.headers, timeout=5)
            if response.status_code == 200:
                raw_text = response.json()[0]["generated_text"]
                answer = raw_text.split("<|assistant|>\nAnswer:")[-1].strip()
                return answer
            else:
                return self.generate_local_fallback(context_chunks)
        except Exception:
            # Instantly switch to internal summary if connection fails
            return self.generate_local_fallback(context_chunks)

    def query(self, user_query, product_filter=None):
        context_chunks, metadata = self.retrieve_context(user_query, product_filter)
        if not context_chunks:
            return "No matching complaint narratives found in the database index.", []
            
        answer = self.generate_answer(user_query, context_chunks)
        
        sources = []
        for doc, meta in zip(context_chunks, metadata):
            sources.append({
                "text": doc,
                "id": meta.get("complaint_id", "N/A"),
                "product": meta.get("product_category", "Unknown")
            })
        return answer, sources