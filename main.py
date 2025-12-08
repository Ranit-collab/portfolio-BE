import os
from fastapi import FastAPI, Query
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from fastapi.responses import JSONResponse
import traceback
from openai import OpenAIError
from fastapi import HTTPException


from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

app = FastAPI(title="Ranit AI Agent")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = "data"
VECTORDB_PATH = "chroma_store"

# -----------------------------
# Load Documents
# -----------------------------
def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)
        if file.endswith(".pdf"):
            docs.extend(PyMuPDFLoader(path).load())
        elif file.endswith(".txt"):
            docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs

def text_splitter(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # system_prompt=["\n\n", "\n", " ", ""]

    )
    split_docs =  text_splitter.split_documents(documents)

    return split_docs

documents = load_documents()
chunks = text_splitter(documents)

# -----------------------------
# Vector Store
# -----------------------------
def create_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    if os.path.exists(VECTORDB_PATH):
        return Chroma(
            persist_directory=VECTORDB_PATH,
            embedding_function=embeddings
        )

    docs = load_documents()

    chunks = text_splitter(docs)

    store = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTORDB_PATH
    )
    store.persist()
    return store

vectorstore = create_vectorstore()

# -----------------------------
# RAG Chain
# -----------------------------
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    timeout=8
)

system_prompt = """
You are Ranit — the person described in the retrieved context.  
You speak in a friendly, confident, and professional tone.

### Your Rules:
1. **Use second-person ("he", "him", "his") but ONLY when the information is explicitly present in the provided context.**
2. If the answer is NOT in the context, reply with exactly:
   "I don’t have enough information about that yet. Please check Ranit's GitHub or LinkedIn, or contact Ranit directly for more details."
3. Never guess, assume, or hallucinate facts about Ranit.
4. Keep responses short, clear, and helpful (2–4 sentences).
5. When describing experience or skills, speak confidently but accurately.
6. If the question is personal (background, education, achievements), answer only using context.
7. If the question is general or technical (coding, frameworks, AI, etc.), answer normally using your own knowledge.

### Your Goals:
- Be a personal guide who explains Ranit’s background, projects, skills, and experiences.
- Provide natural, conversational responses.
- Maintain a warm, professional tone suitable for a portfolio website.

### Context:
{context}
"""


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, qa_chain)

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Ranit AI is running!"}

@app.get("/ask")
async def ask(q: str = Query(...)):
    # empty query protection
    if not q or q.strip() == "":
        return JSONResponse(
            {"error": "Query cannot be empty.", "answer": None},
            status_code=400
        )

    try:
        # Main RAG call
        response = await chain.ainvoke({"input": q})
        answer = response.get("answer")

        # If chain returns nothing
        if not answer:
            answer = "I’m not sure about that. Try rephrasing your question."

        return {
            "question": q,
            "answer": answer
        }

    except OpenAIError as e:
        # OpenAI specific issue → Retry once
        try:
            print("Retrying OpenAI call...")
            response = await chain.ainvoke({"input": q})
            answer = response.get("answer", "Model is currently unavailable. Try again later.")
            return {"question": q, "answer": answer}
        except Exception:
            return JSONResponse(
                {"error": "AI model failed twice.", "answer": "The AI model is currently unavailable."},
                status_code=500
            )

    except Exception as e:
        # Generic fatal error → Never crash API
        print("RAG ERROR:", e)
        print(traceback.format_exc())

        return JSONResponse(
            {
                "error": "Internal server error.",
                "message": str(e),
                "answer": "Something went wrong while processing your question."
            },
            status_code=500
        )

