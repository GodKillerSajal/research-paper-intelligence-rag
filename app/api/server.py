from fastapi import (
    FastAPI,
    UploadFile,
    File
)

import shutil

from app.api.schemas import (
    QuestionRequest
)

from app.rag.pipeline import (
    ingest_document,
    ask_question
)

app = FastAPI()


@app.get("/")
def home():

    return {
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


@app.post("/ask")
def ask(
    request: QuestionRequest
):

    try:

        answer = ask_question(
            request.question
        )

        print("\nANSWER FROM PIPELINE:")
        print(answer)

        return {
            "success": True,
            "answer": answer
        }

    except Exception as e:

        print(f"ERROR: {e}")

        return {
            "success": False,
            "answer": None,
            "error": str(e)
        }


@app.post("/upload")
def upload_pdf(
    file: UploadFile = File(...)
):

    try:
        import os

        print("\nCURRENT WORKING DIRECTORY:")
        print(os.getcwd())

        print("\nUPLOADS EXISTS:")
        print(os.path.exists("uploads"))
        file_path = (
            f"uploads/{file.filename}"
        )

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        num_chunks = ingest_document(
            file_path
        )

        return {
            "success": True,
            "filename": file.filename,
            "chunks": num_chunks,
            "status": "uploaded"
        }

    except Exception as e:

        print(f"UPLOAD ERROR: {e}")

        return {
            "success": False,
            "error": str(e)
        }