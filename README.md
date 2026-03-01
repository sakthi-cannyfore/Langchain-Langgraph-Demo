# 🤖 LangChain RAG Pipeline & LangGraph Multi-Tool Agent

> Two powerful AI projects using **Groq (Free LLM)** + **HuggingFace Embeddings** + **FAISS Vector DB**

---

## 📋 Table of Contents

- [Get Your FREE Groq API Key](https://console.groq.com/home)
- [Create the .env File](#-create-the-env-file)
- [Project 1 — RAG Pipeline](#-project-1--rag-pipeline-langchain)
- [Project 2 — LangGraph Agent](#-project-2--langgraph-multi-tool-agent)
- [Folder Structure](#-folder-structure)

---

## 🔑 Get Your FREE Groq API Key (Step by Step)

Groq gives you **14,400 free requests/day** — no credit card needed.

**Step 1** → Open your browser and go to:
```
https://console.groq.com
```

**Step 2** → Click **"Sign Up"** (top right corner)
- You can sign up with Google, GitHub, or Email
- It's completely free

**Step 3** → After signing in, look at the **left sidebar**
- Click on **"API Keys"**

**Step 4** → Click the **"Create API Key"** button
- Give it a name like `my-rag-project`
- Click **"Submit"**

**Step 5** → Copy the key that appears
```
gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
> ⚠️ **IMPORTANT:** Copy it NOW — Groq only shows it once!

**Step 6** → Paste it into your `.env` file (see next section)

---

## 🗂️ Create the `.env` File

The `.env` file stores your secret API key safely — it never goes into your code.

**Step 1** → Open your project folder in VS Code or File Explorer

**Step 2** → Create a new file named exactly:
```
.env
```
> ⚠️ The dot (.) at the start is required. No `.txt`, no `.env.txt` — just `.env`

**Step 3** → Open the `.env` file and paste this:
```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```
Replace `gsk_xxx...` with your actual key copied from Groq.

**Step 4** → Save the file.

Your `.env` file should look exactly like this:
```
GROQ_API_KEY=gsk_abc123yourkeyhere
```

> ✅ No quotes, no spaces around `=`, just `KEY=value`

---

## 📁 Folder Structure

```
your-project/
│
├── .env                  ← your secret Groq API key (never share this)
├── Langchain.py               ← RAG Pipeline code
├── Langgraph.py              ← LangGraph Agent code
│
└── data/
    └── test.csv          ← your CSV file with Name, Description columns
```

Your `data/test.csv` must look like this:
```csv
Name,Description
Sakthi,"Sakthi is working at Cannyfore Company as a Software Engineer..."
Abi,"Abi is a Machine Learning Developer who is passionate about AI..."
```

---

## 🚀 Project 1 — RAG Pipeline (LangChain)

### What It Does
Reads your CSV file, converts it into embeddings, stores in FAISS vector database,
and answers your questions by finding the most relevant data using cosine similarity.

```
CSV → Pandas → Chunks → Embeddings → FAISS → Question → Groq LLM → Answer
```

### Install Dependencies

Run this once in your terminal:

```bash
pip install langchain langchain-core langchain-community
pip install langchain-text-splitters
pip install langchain-huggingface "huggingface_hub>=0.33.4,<1.0.0"
pip install sentence-transformers faiss-cpu python-dotenv pandas
pip install groq
```

### How the Flow Works

```
Your CSV File
      |
      v  pd.read_csv() — Pandas reads and cleans data
      |
      v  Each row → LangChain Document object
         page_content: "Name: Sakthi | Description: ..."
         metadata:     { "name": "Sakthi" }
      |
      v  RecursiveCharacterTextSplitter
         chunk_size=500, chunk_overlap=100
         Long descriptions split into smaller overlapping pieces
      |
      v  HuggingFace MiniLM (all-MiniLM-L6-v2)
         Each chunk → 384-dimension float vector
         Runs 100% on your CPU — zero cost
      |
      v  FAISS saves all vectors to disk (vector_store/ folder)
      |
      ===================== QUERY TIME =====================
      |
      v  You type a question
      |
      v  MiniLM embeds question → 384-dim vector
      |
      v  FAISS cosine similarity search → top 3 nearest chunks
      |
      v  Prompt built: context (chunks) + question
      |
      v  Groq LLM (LLaMA 3.3 70B) reads prompt → generates answer
      |
      v  response.choices[0].message.content → printed to you
```

### Run It

```bash
py Langchain.py
```

### Example Questions to Ask

```
You: Who is Sakthi?
You: Who uses TensorFlow?
You: Who lives in Coimbatore?
You: Who plays chess?
You: What does Prem work on?
```

### Commands Inside the Chat

| Command  | What It Does                              |
|----------|-------------------------------------------|
| `exit`   | Quit the program                          |
| `reload` | Re-index CSV if you edited the file       |

---

## 🕸️ Project 2 — LangGraph Multi-Tool Agent

### What It Does
A smart agent that decides which tool to use based on your question.
It can search Wikipedia, get real-time weather, and do math calculations —
all while remembering the full conversation history.

```
Question → LLM decides → Tool (Wikipedia / Weather / Calculator) → Answer
```

### Install Dependencies

Run this once in your terminal:

```bash
pip install langgraph langchain-core langchain-groq
pip install requests beautifulsoup4 python-dotenv
```

### Free APIs Used — No Extra Keys Needed

| Tool             | API                                      | API Key? |
|------------------|------------------------------------------|----------|
| Wikipedia Search | `wikipedia.org/wiki/{topic}` scraping    | ❌ None  |
| Weather Report   | `open-meteo.com` (100% free forever)     | ❌ None  |
| Calculator       | Pure Python `math` library               | ❌ None  |
| LLM              | Groq — LLaMA 3.3 70B                     | ✅ Groq  |

### How the Flow Works

```
You type a question
        |
        v
  [AGENT NODE]
  Groq LLM reads question + full chat history
  Decides: do I need a tool?
        |
   YES  |                           NO
        v                            v
  [TOOL NODE]                   Final answer
  Run the chosen tool           printed to you
  Wikipedia / Weather / Calc
        |
  Result added to messages
        |
  Back to [AGENT NODE]
  LLM reads tool result
  Decides: more tools needed? or done?
```

### LangGraph Decision Logic

```
graph.add_conditional_edges("agent", should_continue, {
    "call_tool" : "tools",   ← LLM needs a tool → run it
    "end"       : END        ← LLM has answer   → print it
})
graph.add_edge("tools", "agent")  ← after tool → always back to agent
```

### Memory (How It Remembers)

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # ← grows with every turn
```

Every message (your question, tool calls, tool results, LLM answers)
is added to the list. The LLM receives the full list every time,
so it always knows what was said before.

### Run It

```bash
py Langgraph.py
```

### Example Questions to Ask

```
# Wikipedia tool
You: Who is APJ Abdul Kalam?
You: Tell me about ISRO
You: What is machine learning?

# Weather tool
You: What is the weather in Chennai?
You: Temperature in Tokyo right now
You: Is it raining in London?

# Calculator tool
You: What is 25% of 8500?
You: Calculate sqrt(256)
You: What is 2 to the power of 10?

# Multi-tool (calls 2 tools in one question)
You: Tell me about Chennai and what is the weather there?
You: Who is Elon Musk and calculate his birth year subtracted from 2025

# Memory test (remembers previous messages)
You: Who is Virat Kohli?
You: How old is he?           ← remembers "he" = Virat Kohli
You: What is his age times 2? ← uses memory + calculator
```

### Commands Inside the Chat

| Command  | What It Does                              |
|----------|-------------------------------------------|
| `exit`   | Quit the program                          |
| `clear`  | Reset memory — start fresh conversation   |

---

## ⚡ Quick Start Checklist

```
[ ] 1. Sign up at https://console.groq.com
[ ] 2. Go to API Keys → Create API Key → Copy it
[ ] 3. Create .env file in your project folder
[ ] 4. Paste: GROQ_API_KEY=your_key_here
[ ] 5. Create data/ folder → put test.csv inside
[ ] 6. Run pip install commands for your project
[ ] 7. Run: py Langchain.py   (for RAG Pipeline)
         or: py graph.py  (for LangGraph Agent)
```

---

## ❓ Common Issues

| Problem | Fix |
|---------|-----|
| `GROQ_API_KEY not found` | Check `.env` file exists and has no quotes around the key |
| `ModuleNotFoundError` | Run the pip install commands again |
| `vector_store error` | Run `rmdir /s /q vector_store` then restart |
| `Rate limit 429` | Free tier limit hit — wait a few minutes and retry |
| `City not found` (weather) | Check city name spelling |
| `Wikipedia 404` | Try a more specific topic name |

---

## 🧠 Summary

```
LangChain  = Framework to build AI pipelines (NOT an AI model)
LangGraph  = Framework for AI agents with decision loops (NOT an AI model)
Groq       = Real AI — runs LLaMA 3.3 70B on fast LPU hardware (FREE)
HuggingFace MiniLM = Real AI — converts text to vectors (runs on your CPU)
FAISS      = Vector search library — cosine similarity (NOT an AI)
```

> The intelligence comes from **Groq + HuggingFace**.
> LangChain, LangGraph, and FAISS are just the **wiring** connecting them.

---

*Built with LangChain · LangGraph · Groq · HuggingFace · FAISS · Open-Meteo*