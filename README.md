# 🧠 Resume RAG Chatbot (LangChain + Streamlit)

Upload **many PDF resumes** and ask questions like:
- "Show projects by Priya S"
- "What is Arjun's experience at TCS?"
- "List skills of Kavya"

The bot will **retrieve** the right resume chunks and answer, with **sources (file + page)**.

---

## ✅ What’s inside
- Streamlit app (UI + RAG)
- Local vector DB (**Chroma**) persisted to disk
- Free local **embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- LLM: choose **OpenAI** or **local (Ollama)**

---

## 📁 Folder structure
```
rag-resume-bot/
├─ app.py
├─ requirements.txt
├─ .env.example
├─ storage/
│  ├─ chroma/     # vector DB (auto)
│  └─ uploads/    # your PDF resumes
└─ README.md
```

---

## 🛠️ Setup (Windows/Mac/Linux)

1) Install **Python 3.10+**  
2) Open a terminal in this project folder and run:
```bash
pip install -r requirements.txt
```

3) Configure your LLM:
   - **OpenAI** (simple): copy `.env.example` → `.env`, put your key:
     ```
     OPENAI_API_KEY=YOUR_OPENAI_KEY_HERE
     LLM_PROVIDER=openai
     OPENAI_MODEL=gpt-4o-mini
     ```
   - **Local (free)**: Install **Ollama**, pull a model:
     ```bash
     ollama pull llama3.1:8b
     ```
     Set `.env` like:
     ```
     LLM_PROVIDER=ollama
     OLLAMA_MODEL=llama3.1:8b
     OLLAMA_BASE_URL=http://localhost:11434
     ```

4) Run the app:
```bash
streamlit run app.py
```
Open the local URL shown (usually http://localhost:8501).

---

## ▶️ How to use
1. In the **left sidebar**, upload **multiple PDF resumes**.
2. Click **Ingest PDFs** (builds/updates the vector DB).
3. (Optional) Type a **candidate name** to filter.
4. Ask questions in the main area. See **Sources** for file+page proof.

---

## 🔎 Tips
- **Misspelled names?** We use fuzzy matching (RapidFuzz).
- If a PDF is an **image scan**, you need OCR. Ask us to add OCR if needed.
- You can re-run **Ingest PDFs** anytime to add more resumes.

---

## ❓ Troubleshooting
- If you get rate limits or want free usage, switch to **Ollama**.
- If names aren't detected, the bot still works by content.
- For stronger extraction we can add NER (spaCy) on request.

Enjoy!
