print("MAIN FILE STARTED")
from app.rag.pipeline import (
    ingest_document,
    ask_question
)
def main():
    def main():
        print("INSIDE MAIN")

    answer = ask_question(
        "What was the sample size?"
    )

    print(answer)
if __name__ == "__main__":
    main()