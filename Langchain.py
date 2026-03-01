# ============================================================
#  RAG PIPELINE - Name + Description CSV
#  Uses: Groq (Free + Fast LLM) + FAISS + HuggingFace
#  Flow: CSV -> Pandas -> Chunks -> Embeddings -> FAISS -> LLM
# ============================================================
#
#  INSTALL DEPENDENCIES (run once in terminal):
#
#  pip install langchain langchain-core langchain-community
#  pip install langchain-text-splitters
#  pip install langchain-huggingface "huggingface_hub>=0.33.4,<1.0.0"
#  pip install sentence-transformers faiss-cpu python-dotenv pandas
#  pip install groq
#
#  CREATE a .env file in same folder:
#  GROQ_API_KEY=your_groq_api_key_here
#
#  GET FREE Groq API key at: https://console.groq.com
#
#  PUT your CSV file at: data/test.csv
#  CSV must have columns: Name, Description
# ============================================================

import os
import shutil
import pandas as pd
from dotenv import load_dotenv

# --- Groq SDK (Free + Ultra Fast LLM) ---
from groq import Groq

# --- LangChain components ---
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ============================================================
# CONFIG
# ============================================================
load_dotenv()
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
VECTOR_STORE_PATH = "vector_store"
CSV_PATH          = "data/test.csv"

# Free Groq models (all are free on free tier):
#   llama-3.3-70b-versatile   -> best quality, recommended
#   llama3-8b-8192            -> fastest, lightest
#   mixtral-8x7b-32768        -> large context window
GROQ_MODEL = "llama-3.3-70b-versatile"

# ============================================================
# STEP 1: Load & Parse CSV using Pandas
#
#  Each row is converted into a LangChain Document:
#  page_content: "Name: Sakthi\nDescription: Sakthi is working..."
#  metadata    : { "name": "Sakthi", "row_index": 0 }
# ============================================================
def load_csv(file_path: str):
    print(f"\n📊 Loading CSV: {file_path}")

    df = pd.read_csv(file_path)

    # Clean: drop fully empty rows
    df.dropna(how="all", inplace=True)

    # Clean: fill missing cells with "N/A"
    df.fillna("N/A", inplace=True)

    # Clean: strip whitespace from all string columns
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Validate required columns exist
    required_cols = {"Name", "Description"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required_cols}. Found: {list(df.columns)}")

    # Print professional summary
    print(f"\n{'='*55}")
    print(f"  CSV SUMMARY")
    print(f"{'='*55}")
    print(f"  Total Rows   : {df.shape[0]}")
    print(f"  Total Columns: {df.shape[1]}")
    print(f"  Column Names : {list(df.columns)}")
    print(f"\n  People loaded:")
    for name in df["Name"].tolist():
        print(f"    -> {name}")
    print(f"{'='*55}\n")

    # Convert each row into a LangChain Document
    documents = []
    for idx, row in df.iterrows():
        # Format content naturally so LLM reads it easily
        content = f"Name: {row['Name']}\nDescription: {row['Description']}"
        doc = Document(
            page_content=content,
            metadata={
                "name"      : row["Name"],
                "row_index" : int(idx),
                "source"    : file_path
            }
        )
        
        print(f"how the normal doc look like {doc}")
        documents.append(doc)
        
        print(f"after appent document look like {documents}")

    print(f"Converted {len(documents)} rows into Documents\n")
    return documents, df


# ============================================================
# STEP 2: Split into Chunks
#
#  Long descriptions are split into smaller overlapping
#  pieces so embeddings are accurate and nothing is lost.
#
#  chunk_size=500    -> ~500 characters per chunk
#  chunk_overlap=100 -> 100 char overlap so context is
#                       not cut off at chunk boundaries
# ============================================================
def split_into_chunks(documents):
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"How chunks look like {chunks}")
    print(f"   {len(documents)} documents -> {len(chunks)} chunks\n")
    return chunks


# ============================================================
# STEP 3: Load FREE Local Embedding Model
#
#  all-MiniLM-L6-v2:
#  - Runs 100% on your CPU (zero API cost)
#  - Converts any text -> 384-dimension float vector
#  - normalize_embeddings=True -> enables cosine similarity
# ============================================================
def load_embedding_model():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("   Embedding model ready\n")
    return embeddings


# ============================================================
# STEP 4: Embed Chunks and Store in FAISS
#
#  FAISS stores all vectors locally on disk.
#  At query time:
#    1. Embed the question -> vector
#    2. Cosine similarity against all stored vectors
#    3. Return top-k nearest matching chunks
# ============================================================
def create_vector_store(chunks, embeddings):
    print("Embedding chunks and storing in FAISS...")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    print(f"how the Embedding chunks and storing in FAISS look like {vector_store}")
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"   FAISS vector store saved to '{VECTOR_STORE_PATH}/'\n")
    return vector_store


def load_vector_store(embeddings):
    print("Loading existing FAISS vector store...")
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("   Vector store loaded\n")
    return vector_store


# ============================================================
# STEP 5: Setup Groq Client (Free + Ultra Fast)
#
#  Groq runs LLaMA and Mixtral models on custom LPU hardware.
#  It is the fastest free LLM API available.
#  Free tier: 14,400 requests/day, no credit card needed.
# ============================================================
def load_groq_client():
    print(f"Loading Groq client (model: {GROQ_MODEL})...")
    client = Groq(api_key=GROQ_API_KEY)
    print("   Groq client ready\n")
    return client


# ============================================================
# STEP 6: Retrieve Relevant Chunks from FAISS
#
#  Embeds the question, searches FAISS using cosine
#  similarity, returns top-k most relevant chunks.
# ============================================================
def retrieve_context(question: str, vector_store, k: int = 3) -> str:
    # similarity_search embeds question and finds nearest vectors
    docs = vector_store.similarity_search(question, k=k)

    # Format each chunk clearly for the LLM prompt
    formatted = []
    for i, doc in enumerate(docs, 1):
        name = doc.metadata.get("name", "Unknown")
        formatted.append(f"[Match {i} - {name}]\n{doc.page_content}")

    return "\n\n".join(formatted)


# ============================================================
# STEP 7: Call Groq LLM with Context + Question
#
#  Full RAG flow when you ask a question:
#
#  Your Question
#       |
#       v
#  [Embedding Model] -> converts question to 384-dim vector
#       |
#       v
#  [FAISS] -> cosine similarity search -> top 3 chunks
#       |
#       v
#  [Prompt] -> context (chunks) + question combined
#       |
#       v
#  [Groq LLM - LLaMA 3.3 70B] -> reads context, answers
#       |
#       v
#  response.choices[0].message.content -> printed to terminal
# ============================================================
def ask_llm(question: str, context: str, client) -> str:
    prompt = f"""You are a helpful assistant with access to a people directory.
Answer the question using ONLY the context provided below.
If the answer is not found in the context, say "I don't have that information."
Be clear, concise, and accurate.

Context:
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role"   : "system",
                "content": "You are a helpful assistant that answers questions based only on the provided context."
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1024
    )

    return response.choices[0].message.content


# ============================================================
# MAIN - Entry Point
# ============================================================
def main():
    print("\n" + "="*55)
    print("   RAG PIPELINE - People Directory Q&A")
    print(f"   LLM  : Groq ({GROQ_MODEL})")
    print(f"   Embed: sentence-transformers/all-MiniLM-L6-v2")
    print(f"   DB   : FAISS (local)")
    print("="*55)

    # Load embedding model (needed for both create and load)
    embeddings = load_embedding_model()

    # Check for actual FAISS index file (not just the folder)
    faiss_index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")

    if not os.path.exists(faiss_index_file):
        # First run: process CSV from scratch
        print("No vector store found. Processing CSV...\n")
        documents, df = load_csv(CSV_PATH)
        chunks        = split_into_chunks(documents)
        vector_store  = create_vector_store(chunks, embeddings)
    else:
        # Subsequent runs: load saved FAISS (much faster)
        print("Existing vector store found. Skipping re-processing.\n")
        vector_store = load_vector_store(embeddings)

    # Load Groq LLM client
    client = load_groq_client()

    print("="*55)
    print("  System Ready! Ask anything about the people.")
    print("")
    print("  Commands:")
    print("    'exit'   -> quit the program")
    print("    'reload' -> re-index CSV after you edit it")
    print("="*55 + "\n")

    # Interactive chat loop
    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue

        # Quit command
        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Reload command: delete old FAISS index and rebuild
        if query.lower() == "reload":
            print("\nReloading CSV and rebuilding vector store...")
            shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            documents, df = load_csv(CSV_PATH)
            chunks        = split_into_chunks(documents)
            vector_store  = create_vector_store(chunks, embeddings)
            print("Reload complete!\n")
            continue

        # Step 1: Embed question + search FAISS (cosine similarity)
        context = retrieve_context(query, vector_store, k=3)

        # Step 2: Send context + question to Groq LLM
        print("\nLLM: ", end="", flush=True)
        response = ask_llm(query, context, client)
        print(response)
        print("-" * 55 + "\n")


if __name__ == "__main__":
    main()