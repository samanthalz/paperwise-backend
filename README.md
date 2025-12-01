# 📄 PaperWise Backend API

**Smart CS Paper Analyser — FastAPI Backend**

The **PaperWise Backend** powers the intelligent research-analysis engine behind the PaperWise platform.
It exposes REST APIs for paper processing, embeddings, recommendations, and RAG-powered Q&A — all consumed by the Next.js frontend.

---

## 📌 Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Tech Stack](#tech-stack)
* [Architecture](#architecture)
* [Project Structure](#project-structure)
* [Copyright](#copyright)

---

## 🔍 Overview

The PaperWise backend is a modular FastAPI service that orchestrates:

* Research paper ingestion and processing
* Chunking + embedding workflows
* Vector database operations
* Recommendation system execution
* Q&A with retrieval-augmented generation (RAG)
* User, metadata, and storage management with Supabase

It integrates multiple ML/AI and data APIs to deliver a seamless research-assistant experience.

---

## ✨ Features

* 🧩 **Modular REST API architecture**
* 🔍 **Paper processing pipeline** (Docling + PDF extraction + layout analysis)
* 🧠 **Embedding generation** using Specter2-Base, EmbeddingGemma-300M, and multi-qa-mpnet-base-dot-v1
* 🔎 **Semantic search & vector retrieval** (Supabase Vector Store)
* 🤖 **RAG-based question answering** using Gemini 2.5 Flash Preview
* 📡 **Live data ingestion** from ArXiv & Semantic Scholar
* 📁 **Supabase Storage integration** (PDFs, thumbnails)
* 📊 **Dataset evaluation** using SPIQA
* 🔐 **User data & metadata persistence** (Supabase PostgreSQL)

---

## 🛠 Tech Stack

### **Backend Framework**

* FastAPI (Python)

### **Database & Storage**

* Supabase (PostgreSQL + Vector Store)
* Supabase Storage (PDFs, thumbnails)

### **ML / AI / Research APIs**

* ArXiv API
* Semantic Scholar API
* Gemini 2.5 Flash Preview API
* Docling
* Specter2-Base
* EmbeddingGemma-300M
* multi-qa-mpnet-base-dot-v1
* SPIQA dataset evaluator

### **Libraries**

* supabase-py
* PyMuPDF
* Pydantic
* Uvicorn

---

## 🏗 Architecture

The system architecture consists of four interconnected layers:

### **1️⃣ Frontend (Next.js)**

* User interaction layer
* Displays papers, recommendations, and interactive chat
* Sends all requests (paper details, Q&A, search, recommendations) to the backend

### **2️⃣ Backend (FastAPI — This repo)**

Functions as the **orchestrator** of the entire system:

* Ingests and processes research papers (PDF → structured → chunks → embeddings)
* Runs the hybrid recommendation engine
* Manages vector search flow for Q&A
* Invokes external APIs (ArXiv, Gemini, Docling, Semantic Scholar)
* Sends structured results back to the frontend

### **3️⃣ Database & Storage (Supabase)**

* **PostgreSQL Database**
  Stores:
  * Users
  * Paper metadata
  * Processed chunk metadata
  * Notes, folders, user-generated content

* **Vector Store**
  Contains all embeddings for fast semantic lookup
  
* **Supabase Storage**
  Holds:
  * Raw PDFs
  * Paper thumbnails

### **4️⃣ External Services**

* ArXiv → Raw paper source
* Docling → PDF parsing, layout detection, figure/table extraction
* Semantic Scholar → Metadata enrichment
* Gemini → Final RAG reasoning for Q&A
---

## 📁 Project Structure

```
paperwise-backend/
│── app/
│   ├── api/                # All route controllers
│   ├── services/           # Logic for papers, embeddings, RAG, etc.
│   ├── db/                 # Supabase client, DB utils
│   ├── models/             # Pydantic schemas
│   ├── utils/              # Pdf parsing, helpers
│   ├── main.py             # FastAPI entrypoint
│── tests/                  # SPIQA tests
│── requirements.txt
│── README.md
```
---

## 📜 Copyright

Copyright (c) 2025 PaperWise / samanthalz

All rights reserved.

This source code and its associated files are proprietary and confidential.
Unauthorized copying, distribution, modification, or use of this code, via
any medium, is strictly prohibited.

No license is granted to use, distribute, or modify this software unless
explicit written permission is obtained from the copyright holder.

