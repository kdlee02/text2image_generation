import os
import shutil

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings

load_dotenv()

# Resolved from this file, not the process working directory: app/ imports this
# module while running with cwd=app/, and the old "./vectorstore" default created
# (and deleted) a stray index wherever the process happened to start.
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectorstore")

# A prompt is worth showing the evaluator only if it scored well AND the loop had
# actually rewritten it — those are the examples of a *good revision*.
MIN_SCORE = 8


class DocumentProcessor:
    """Retrieves prompts that previously scored well, to use as few-shot examples."""

    def __init__(self, persist_directory=DEFAULT_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embedding_model = UpstageEmbeddings(
            api_key=os.getenv("UPSTAGE_API_KEY"),
            model="embedding-query",
        )
        self.vectorstore = None
        self.load_vectorstore()

    def load_vectorstore(self):
        """Load a persisted index if one exists."""
        if not os.path.exists(self.persist_directory):
            return
        try:
            self.vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embedding_model,
                # Required by FAISS.load_local; it unpickles the docstore. Safe here
                # because we only ever load an index this app built itself. The index
                # is gitignored so no pickle ships to anyone cloning the repo.
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            print(f"[rag] could not load vectorstore: {e}")
            self.vectorstore = None

    def save_vectorstore(self):
        if self.vectorstore is not None:
            self.vectorstore.save_local(self.persist_directory)

    @staticmethod
    def load_successful_prompts(csv_path):
        """Prompts the loop revised into something that scored above MIN_SCORE."""
        df = pd.read_csv(csv_path)
        df = df[pd.to_numeric(df["overall_score"], errors="coerce") > MIN_SCORE]
        df = df[df["original_user_prompt"] != df["prompt"]]
        prompts = df["prompt"].dropna().drop_duplicates()
        return [Document(page_content=str(p)) for p in prompts]

    def process_document(self, csv_path, rebuild=False):
        """Build the index from a log CSV. Returns True if it (re)built.

        The previous version early-returned whenever an index existed, so every
        later upload was a silent no-op that still reported success.
        """
        if self.vectorstore is not None and not rebuild:
            return False
        docs = self.load_successful_prompts(csv_path)
        if not docs:
            print(f"[rag] no prompts in {csv_path} scored above {MIN_SCORE}; index not built")
            return False
        # No text splitter: these prompts average ~360 characters, so chunking at
        # 1000 was a no-op that just added a dependency.
        self.vectorstore = FAISS.from_documents(docs, self.embedding_model)
        self.save_vectorstore()
        print(f"[rag] indexed {len(docs)} successful prompts")
        return True

    def retrieve_prompts(self, query, k=3):
        """Top-k similar successful prompts, as plain strings."""
        if self.vectorstore is None:
            return []
        return [d.page_content.strip() for d in self.vectorstore.similarity_search(query, k=k)]

    def reset(self):
        """Drop the index from memory and disk."""
        self.vectorstore = None
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
