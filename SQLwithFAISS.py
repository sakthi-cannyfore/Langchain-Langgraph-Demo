# ============================================================
#  RAG PIPELINE - Relational Database (SQLite) + FAISS + Groq
#
#  Tables:
#    users        -> id, name, email, department, role
#    descriptions -> id, user_id (FK), bio, skills, location
#    appraisals   -> id, user_id (FK), rating, review, year
#
#  Flow: SQLite (JOIN all tables) -> Documents -> Chunks
#        -> Embeddings -> FAISS -> Groq LLM -> Answer
#
#  INSTALL:
#    pip install langchain langchain-core langchain-community
#    pip install langchain-text-splitters
#    pip install langchain-huggingface "huggingface_hub>=0.33.4,<1.0.0"
#    pip install sentence-transformers faiss-cpu python-dotenv
#    pip install groq
#
#  .env file:
#    GROQ_API_KEY=your_groq_api_key_here
# ============================================================

import os
import sqlite3
import shutil
from dotenv import load_dotenv
from groq import Groq
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
DB_PATH           = "data/people.db"
GROQ_MODEL        = "llama-3.3-70b-versatile"

# ============================================================
# STEP 0: Create SQLite Database with 3 Relational Tables
#
#  users        (id PK)
#  descriptions (id PK, user_id FK -> users.id)
#  appraisals   (id PK, user_id FK -> users.id)
#
#  Sample data is inserted so you can run immediately.
# ============================================================
def setup_database():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # ── Create Tables ──────────────────────────────────────
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            name       TEXT NOT NULL,
            email      TEXT UNIQUE,
            department TEXT,
            role       TEXT
        );

        CREATE TABLE IF NOT EXISTS descriptions (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            bio     TEXT,
            skills  TEXT,
            location TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS appraisals (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            year    INTEGER,
            rating  REAL,
            review  TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # ── Insert Sample Data (only if tables are empty) ──────
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        sample_users = [
            ("Sakthi",  "sakthi@company.com",  "Engineering",  "Senior Developer"),
            ("Priya",   "priya@company.com",   "Marketing",    "Marketing Manager"),
            ("Arjun",   "arjun@company.com",   "Engineering",  "DevOps Engineer"),
            ("Meena",   "meena@company.com",   "HR",           "HR Specialist"),
            ("Vikram",  "vikram@company.com",  "Finance",      "Financial Analyst"),
        ]
        cur.executemany(
            "INSERT INTO users (name, email, department, role) VALUES (?,?,?,?)",
            sample_users
        )

        sample_descriptions = [
            (1, "Sakthi is a passionate developer with 8 years of experience in Python and AI systems.",
                "Python, FastAPI, LangChain, Docker, PostgreSQL", "Chennai"),
            (2, "Priya leads the marketing team with a focus on digital campaigns and brand growth.",
                "SEO, Google Ads, Content Strategy, Analytics", "Bangalore"),
            (3, "Arjun manages cloud infrastructure and CI/CD pipelines for the engineering team.",
                "AWS, Kubernetes, Terraform, Jenkins, Linux", "Hyderabad"),
            (4, "Meena handles recruitment, employee relations, and performance management.",
                "Recruitment, HRIS, Labour Law, Conflict Resolution", "Mumbai"),
            (5, "Vikram analyzes financial data and builds forecasting models for business decisions.",
                "Excel, Power BI, Financial Modelling, SQL", "Delhi"),
        ]
        cur.executemany(
            "INSERT INTO descriptions (user_id, bio, skills, location) VALUES (?,?,?,?)",
            sample_descriptions
        )

        sample_appraisals = [
            # (user_id, year, rating, review)
            (1, 2023, 4.8, "Sakthi delivered the AI pipeline project ahead of schedule. Exceptional problem-solving skills."),
            (1, 2024, 4.9, "Led the RAG implementation team. Mentored 3 junior developers. Highly recommended for promotion."),
            (2, 2023, 4.2, "Priya ran a successful Q4 campaign resulting in 35% traffic increase. Great team player."),
            (2, 2024, 4.5, "Executed brand refresh strategy. Exceeded all KPIs for the year."),
            (3, 2023, 4.6, "Arjun reduced deployment time by 60% with new CI/CD pipelines. Excellent technical skills."),
            (3, 2024, 4.7, "Migrated entire infrastructure to Kubernetes. Zero downtime achieved."),
            (4, 2023, 4.0, "Meena improved onboarding process and reduced time-to-hire by 20%."),
            (4, 2024, 4.3, "Implemented new HRIS system successfully. Strong stakeholder management."),
            (5, 2023, 4.4, "Vikram built revenue forecasting model that improved accuracy by 25%."),
            (5, 2024, 4.6, "Delivered board-level financial dashboards. Recognized as top performer in Finance."),
        ]
        cur.executemany(
            "INSERT INTO appraisals (user_id, year, rating, review) VALUES (?,?,?,?)",
            sample_appraisals
        )

        conn.commit()
        print("Sample data inserted into database.")

    conn.close()
    print(f"Database ready: {DB_PATH}\n")


# ============================================================
# STEP 1: Load Data via SQL JOIN (all 3 tables combined)
#
#  Joins users + descriptions + appraisals on user_id.
#  Each row = one person's full info from all 3 tables.
#
#  Each person is converted into a LangChain Document:
#  page_content = rich text combining all table data
#  metadata     = name, user_id, department, year
# ============================================================
def load_from_database() -> list[Document]:
    print(f"Loading data from SQLite: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # access columns by name
    cur = conn.cursor()

    # ── JOIN all 3 tables ───────────────────────────────────
    query = """
        SELECT
            u.id         AS user_id,
            u.name,
            u.email,
            u.department,
            u.role,
            d.bio,
            d.skills,
            d.location,
            a.year       AS appraisal_year,
            a.rating,
            a.review
        FROM users u
        LEFT JOIN descriptions d ON d.user_id = u.id
        LEFT JOIN appraisals   a ON a.user_id = u.id
        ORDER BY u.id, a.year
    """
    rows = cur.execute(query).fetchall()
    conn.close()

    for i, row in enumerate(rows, start=1):
        print(f"Row {i}: {row}")

    print(f"Fetched {(rows)} rows from JOIN query.\n")

    # ── Group appraisals per user ───────────────────────────
    # One user can have multiple appraisal years -> group them
    from collections import defaultdict
    user_data   = {}
    user_apprs  = defaultdict(list)

    for row in rows:
        uid = row["user_id"]
        if uid not in user_data:
            user_data[uid] = {
                "user_id"   : uid,
                "name"      : row["name"],
                "email"     : row["email"],
                "department": row["department"],
                "role"      : row["role"],
                "bio"       : row["bio"]       or "N/A",
                "skills"    : row["skills"]    or "N/A",
                "location"  : row["location"]  or "N/A",
            }
        if row["appraisal_year"]:
            user_apprs[uid].append({
                "year"  : row["appraisal_year"],
                "rating": row["rating"],
                "review": row["review"],
            })

    # ── Build Documents ─────────────────────────────────────
    documents = []
    for uid, info in user_data.items():
        apprs = user_apprs[uid]
     
        # Format appraisals as readable text
        appr_text = ""
        if apprs:
            for a in apprs:
                appr_text += (
                    f"\n  [{a['year']}] Rating: {a['rating']}/5.0\n"
                    f"  Review: {a['review']}"
                )
        else:
            appr_text = "\n  No appraisal records found."

        # Rich content block - LLM reads this as context
        content = f"""
=== EMPLOYEE PROFILE ===
Name      : {info['name']}
Email     : {info['email']}
Department: {info['department']}
Role      : {info['role']}
Location  : {info['location']}

--- DESCRIPTION ---
Bio       : {info['bio']}
Skills    : {info['skills']}

--- APPRAISAL HISTORY ---
{appr_text}
""".strip()

        doc = Document(
            page_content=content,
            metadata={
                "name"       : info["name"],
                "user_id"    : uid,
                "department" : info["department"],
                "role"       : info["role"],
                "source"     : "sqlite_db",
            }
        )
        documents.append(doc)
        print(f"  Document created for: {info['name']} ({len(apprs)} appraisal records)")

    print(f"\nTotal Documents: {len(documents)}\n")
    return documents


# ============================================================
# STEP 2: Split into Chunks
# ============================================================
# from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_into_chunks(documents: list[Document]) -> list[Document]:
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"  {len(documents)} documents -> {len(chunks)} chunks\n")
    return chunks


# ============================================================
# STEP 3: Load Embedding Model
# ============================================================
def load_embedding_model():
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("  Embedding model ready\n")
    return embeddings


# ============================================================
# STEP 4: Create / Load FAISS Vector Store
# ============================================================
def create_vector_store(chunks, embeddings):
    print("Embedding chunks into FAISS...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"  FAISS saved to '{VECTOR_STORE_PATH}/'\n")
    return vector_store


def load_vector_store(embeddings):
    print("Loading existing FAISS vector store...")
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("  Vector store loaded\n")
    return vector_store


# ============================================================
# STEP 5: Load Groq LLM Client
# ============================================================
def load_groq_client():
    print(f"Loading Groq client (model: {GROQ_MODEL})...")
    client = Groq(api_key=GROQ_API_KEY)
    print("  Groq client ready\n")
    return client


# ============================================================
# STEP 6: Retrieve Relevant Chunks from FAISS
# ============================================================
def retrieve_context(question: str, vector_store, k: int = 4) -> str:
    docs = vector_store.similarity_search(question, k=k)
    formatted = []
    for i, doc in enumerate(docs, 1):
        name = doc.metadata.get("name", "Unknown")
        dept = doc.metadata.get("department", "")
        formatted.append(f"[Match {i} - {name} | {dept}]\n{doc.page_content}")
    return "\n\n".join(formatted)


# ============================================================
# STEP 7: Ask Groq LLM
#
#  The LLM receives:
#    - System prompt (role: HR assistant with DB access)
#    - Context (joined data from all 3 tables via FAISS)
#    - User question
#
#  This is how LLMs "access" relational data:
#    SQL JOIN -> rich text document -> embeddings -> FAISS
#    -> top-k chunks -> LLM reads as context -> answers
# ============================================================
def ask_llm(question: str, context: str, client) -> str:
    prompt = f"""You are an intelligent HR assistant with access to an employee database.
The database has 3 related tables:
  1. users        - name, email, department, role
  2. descriptions - bio, skills, location
  3. appraisals   - yearly rating and review per employee

Answer the question using ONLY the context below.
If an answer is not in the context, say "I don't have that information."
Be precise and reference specific data (ratings, skills, years) when available.

Context from Database (all tables joined):
{context}

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role"   : "system",
                "content": (
                    "You are an HR assistant. Answer only from the provided context. "
                    "Always reference specific facts like ratings, years, skills when available."
                )
            },
            {
                "role"   : "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=1024
    )
    return response.choices[0].message.content


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*60)
    print("  RAG PIPELINE - Relational DB (SQLite) + FAISS + Groq")
    print("  Tables: users | descriptions | appraisals (FK joined)")
    print(f"  LLM   : {GROQ_MODEL}")
    print(f"  Embed : all-MiniLM-L6-v2")
    print(f"  DB    : SQLite ({DB_PATH})")
    print("="*60 + "\n")

    # Setup DB and insert sample data if needed
    setup_database()

    # Load embeddings
    embeddings = load_embedding_model()

    # Check if FAISS index exists
    faiss_index_file = os.path.join(VECTOR_STORE_PATH, "index.faiss")
    if not os.path.exists(faiss_index_file):
        print("No vector store found. Building from database...\n")
        documents    = load_from_database()
        chunks       = split_into_chunks(documents)
        vector_store = create_vector_store(chunks, embeddings)
    else:
        print("Existing vector store found. Skipping rebuild.\n")
        vector_store = load_vector_store(embeddings)

    # Load LLM
    client = load_groq_client()

    print("="*60)
    print("  System Ready! Ask anything about employees.")
    print("")
    print("  Example questions:")
    print("    - What are Sakthi's skills?")
    print("    - Who got the highest appraisal rating in 2024?")
    print("    - What department is Priya in?")
    print("    - Tell me about Arjun's appraisal history")
    print("    - Who works in Engineering?")
    print("")
    print("  Commands:")
    print("    'exit'   -> quit")
    print("    'reload' -> re-sync from database and rebuild FAISS")
    print("="*60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if query.lower() == "reload":
            print("\nRebuilding vector store from database...")
            shutil.rmtree(VECTOR_STORE_PATH, ignore_errors=True)
            documents    = load_from_database()
            chunks       = split_into_chunks(documents)
            vector_store = create_vector_store(chunks, embeddings)
            print("Reload complete!\n")
            continue

        # RAG: retrieve context from FAISS -> ask LLM
        context  = retrieve_context(query, vector_store, k=5)
        print("\nLLM: ", end="", flush=True)
        response = ask_llm(query, context, client)
        print(response)
        print("-" * 60 + "\n")


if __name__ == "__main__":
    main()
