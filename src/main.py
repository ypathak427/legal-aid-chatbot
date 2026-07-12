import os
import time
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from google import genai
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
BASE_DIR = Path(__file__).resolve().parent.parent  # goes up from src/ to project root
DB_DIR = BASE_DIR / "legal_db_v3"  # includes BNS + BNSS
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# --- Load vector store + embedding model once at startup ---
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory=str(DB_DIR),
    embedding_function=embedding_model
)

# --- Gemini client ---
client = genai.Client(api_key=GEMINI_API_KEY)

# --- FastAPI app ---
app = FastAPI(title="Legal Aid Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    question: str
    k: int = 4

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


def detect_act_filter(query: str):
    """
    Detects if the query explicitly mentions a specific act, so we can restrict
    retrieval to that act only. Also doubles as a strong signal that the question
    is legal in nature, letting us skip the extra LLM classification call below.
    """
    query_lower = query.lower()

    if "bnss" in query_lower or "nagarik suraksha" in query_lower:
        return "bnss"
    elif "bns" in query_lower or "nyaya sanhita" in query_lower:
        return "bns"
    elif "rti" in query_lower or "right to information" in query_lower:
        return "rti"
    elif "consumer" in query_lower:
        return "consumer"
    elif "crpc" in query_lower or "criminal procedure" in query_lower:
        return "crpc"
    elif "ipc" in query_lower or "penal code" in query_lower:
        return "ipc"
    return None


def generate_with_retry(prompt: str, model: str = "gemini-2.5-flash", max_retries: int = 3):
    """
    Calls Gemini with exponential backoff retry on transient server errors (503, 500).
    """
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response
        except genai_errors.ServerError as e:
            if attempt == max_retries:
                raise
            print(f"[Retry {attempt}/{max_retries}] Gemini server error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2


def is_legal_question(query: str) -> bool:
    """
    Quick classification step for AMBIGUOUS questions only (no act keyword matched).
    This costs one extra Gemini API call, so we only use it as a fallback —
    skipped entirely when detect_act_filter() already found an explicit act mention.
    """
    classification_prompt = f"""Answer with ONLY one word: YES or NO.

Is the following question related to Indian law, legal rights, criminal procedure,
the Indian Penal Code (IPC), Bharatiya Nyaya Sanhita (BNS), Code of Criminal Procedure (CrPC),
Bharatiya Nagarik Suraksha Sanhita (BNSS), Right to Information (RTI), or Consumer Protection?

Question: "{query}"

Answer (YES or NO only):"""

    try:
        response = generate_with_retry(classification_prompt, max_retries=1)
        answer = response.text.strip().upper()
        return answer.startswith("YES")
    except Exception:
        return True


@app.get("/")
def root():
    return {"status": "Legal Aid Chatbot API is running"}


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    act_filter = detect_act_filter(request.question)

    # --- Guardrail: only run the extra LLM classification call if no act
    # was explicitly named. If detect_act_filter() already matched something
    # (e.g. "IPC", "RTI"), we already know it's a legal question — skip the
    # extra API call entirely to save quota. ---
    if act_filter is None:
        if not is_legal_question(request.question):
            return QueryResponse(
                answer="I'm a legal aid assistant focused on Indian law — specifically the IPC/BNS, CrPC/BNSS, RTI Act, and Consumer Protection Act. This question doesn't seem related to those topics, so I'm not able to help with it here. Feel free to ask me something about your legal rights instead!",
                sources=[]
            )

    if act_filter:
        results = vectorstore.similarity_search(
            request.question, k=request.k, filter={"source_act": act_filter}
        )
    else:
        results = vectorstore.similarity_search(request.question, k=request.k)

    context_parts = []
    source_list = []
    for doc in results:
        act = doc.metadata.get("source_act", "unknown").upper()
        page = doc.metadata.get("page_label", "?")
        context_parts.append(f"[Source: {act}, Page {page}]\n{doc.page_content}")
        source_list.append(f"{act}, Page {page}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a legal aid assistant helping Indian citizens understand their legal rights and options.
Answer the question using ONLY the context below. If the context doesn't fully answer the question, say so clearly.
Always mention which Act and Section your answer is based on. Keep the tone clear and non-intimidating for a layperson.

IMPORTANT — CURRENT LAW NOTICE:
As of July 1, 2024, the Indian Penal Code (IPC) has been replaced by the Bharatiya Nyaya
Sanhita (BNS), and the Code of Criminal Procedure (CrPC) has been replaced by the Bharatiya
Nagarik Suraksha Sanhita (BNSS). If your answer is based on IPC or CrPC content, clearly add
a note telling the user that this provision may now fall under BNS or BNSS instead, and
recommend they verify the corresponding current section if it matters for their situation.
If the retrieved context includes both an old (IPC/CrPC) and new (BNS/BNSS) provision on the
same topic, mention both and clarify which one is currently in force.

CONTEXT:
{context}

QUESTION:
{request.question}

ANSWER:"""

    try:
        response = generate_with_retry(prompt)
        answer_text = response.text
    except genai_errors.ServerError:
        answer_text = "Sorry, the AI service is temporarily overloaded. Please try again in a minute."

    return QueryResponse(answer=answer_text, sources=source_list)