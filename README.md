# Intelligent Complaint Analysis for Financial Services
### 🤖 RAG-Powered Analytics Agent for CrediTrust Financial

---

## 📋 Overview
CrediTrust Financial is a digital finance provider serving East African markets via a mobile-first framework spanning **Credit Cards, Personal Loans, Savings Accounts, and Money Transfers**. With over 500,000 active users, manual parsing of thousands of monthly multi-channel complaint logs creates severe analytical bottlenecks.

This project delivers a **Retrieval-Augmented Generation (RAG)** pipeline that transforms raw, unstructured customer complaint narratives into a semantic search database. This enables stakeholders (such as Product Managers, Compliance Teams, and Support leads) to query bulk feedback using plain English and extract synthesized, data-grounded insights in real-time.

---

## 📂 Project Architecture
```text
rag-complaint-chatbot/
├── data/
│   ├── raw/                 # Raw CFPB complaints data source (Ignored by Git)
│   └── processed/           # Filtered base stacks & sampling blocks
├── src/
│   ├── preprocess.py        # Task 1: NLP cleaning, regex, and product parsing
│   └── chunk_embed.py       # Task 2: Stratified chunking & offline database ingestion
├── vector_store/            # Local air-gapped persistent ChromaDB storage files
├── interim_report.md        # Comprehensive academic technical report
├── requirements.txt         # Core environment dependencies
└── README.md                # System configuration and interface guide