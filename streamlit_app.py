# Streamlit Community Cloud Root Entrypoint
# Points directly to the main frontend application
import runpy

if __name__ == "__main__":
    runpy.run_path("frontend/streamlit_app.py", run_name="__main__")
