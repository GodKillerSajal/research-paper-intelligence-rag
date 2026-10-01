import requests
import streamlit as st

st.title("Research Paper RAG")

# 1. Define the file uploader widget
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# 2. Check if a file has been uploaded
if uploaded_file is not None:
    st.success("PDF Selected")

    # 3. Trigger the upload when the button is clicked
    if st.button("Upload"):
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
        }

        with st.spinner("Uploading..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/upload", files=files
                )

                if response.status_code == 200:
                    st.json(response.json())
                else:
                    st.error(
                        f"Upload failed with status code: {response.status_code}"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the backend server. Is your FastAPI/Backend running on port 8000?"
                )
    question = st.text_input(
        "Ask a Question"
            )
    if st.button(
    "Ask"
    ):
        response = requests.post(
        "http://127.0.0.1:8000/ask",
        json={
            "question": question
        }
    )
        st.write(
    response.json()["answer"]
    )