import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Research Paper Intelligence",
    page_icon="📚",
    layout="wide",
)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("📚 Research Intelligence")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Query", "📂 Collections", "🔍 Integrity Analysis", "📊 Evaluation"],
)


def api_get(path):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is FastAPI running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path, json_data=None, files=None, data=None):
    try:
        r = requests.post(
            f"{API_URL}{path}",
            json=json_data,
            files=files,
            data=data,
            timeout=180,
        )
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to backend. Is FastAPI running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def get_collections():
    data = api_get("/collections")
    if data:
        return [c["name"] for c in data]
    return []


# ─── PAGE: Query ──────────────────────────────────────────────────────────────
if page == "🏠 Query":
    st.title("🔍 Research Paper Q&A")

    col1, col2 = st.columns([2, 1])
    with col1:
        collections = get_collections()
        if not collections:
            st.warning("No collections found. Go to **Collections** to create one and upload papers.")
            st.stop()
        selected_collection = st.selectbox("📂 Select Collection", collections)

    with col2:
        retrieval_mode = st.selectbox(
            "🔎 Retrieval Mode",
            ["hybrid", "vector", "bm25"],
            help="hybrid = vector + BM25 + reranker (recommended)",
        )
        top_k = st.slider("Top K sources", 1, 10, 5)

    question = st.text_area(
        "💬 Ask a question about the research papers:",
        height=100,
        placeholder="e.g. What was the sample size used in the experiment?",
    )

    if st.button("🚀 Ask", type="primary") and question.strip():
        with st.spinner("Retrieving and generating answer..."):
            result = api_post("/query", json_data={
                "question": question,
                "collection": selected_collection,
                "top_k": top_k,
                "retrieval_mode": retrieval_mode,
            })

        if result:
            conf = result.get("confidence", "unknown")
            conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴", "insufficient": "⚫"}.get(conf, "⚪")
            st.markdown(f"**Confidence:** {conf_color} `{conf.upper()}`")

            st.markdown("### 📝 Answer")
            answer_text = result.get("answer", "")
            if answer_text.startswith("INSUFFICIENT_EVIDENCE"):
                st.warning(f"⚠️ {answer_text}")
            else:
                st.markdown(answer_text)

            sources = result.get("sources", [])
            if sources:
                st.markdown("### 📌 Sources")
                for src in sources:
                    with st.expander(
                        f"[{src['index']}] {src['document']} — Page {src['page']} · {src['section']}"
                    ):
                        st.markdown(f"```\n{src['text_snippet']}\n```")

            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Retrieval", f"{result['retrieval_latency_ms']:.0f} ms")
            m2.metric("Generation", f"{result['generation_latency_ms']:.0f} ms")
            m3.metric("Total", f"{result['total_latency_ms']:.0f} ms")


# ─── PAGE: Collections ────────────────────────────────────────────────────────
elif page == "📂 Collections":
    st.title("📂 Collections & Documents")

    with st.expander("➕ Create New Collection"):
        col_name = st.text_input("Collection Name", placeholder="e.g. transformer-research")
        col_desc = st.text_input("Description", placeholder="Papers about transformer architectures")
        if st.button("Create Collection"):
            result = api_post("/collections", json_data={"name": col_name, "description": col_desc})
            if result:
                st.success(f"✅ Collection '{col_name}' created!")
                st.rerun()

    st.markdown("### 📤 Upload Paper")
    collections = get_collections()
    if collections:
        upload_collection = st.selectbox("Upload to Collection", collections, key="upload_col")
        chunk_size = st.select_slider(
            "Chunk Size (words)",
            options=[128, 256, 300, 512, 768],
            value=300,
            help="300 words recommended — tighter chunks improve retrieval precision",
        )
        uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])
        if uploaded_file and st.button("📤 Upload & Ingest", type="primary"):
            with st.spinner(f"Parsing, chunking (size={chunk_size}), embedding and indexing..."):
                result = api_post(
                    "/documents/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    data={"collection": upload_collection, "chunk_size": str(chunk_size)},
                )
            if result:
                st.success(
                    f"✅ **{result['filename']}** ingested! "
                    f"{result['num_chunks']} chunks across {result['num_pages']} pages."
                )
    else:
        st.info("Create a collection first.")

    st.markdown("### 📋 All Collections")
    all_collections = api_get("/collections") or []
    for col in all_collections:
        with st.expander(
            f"📂 {col['name']} — {col['document_count']} vectors | {col.get('description', '')}"
        ):
            docs = api_get(f"/documents/{col['name']}")
            if docs:
                for doc in docs:
                    st.markdown(
                        f"- 📄 **{doc['filename']}** — {doc['num_pages']} pages, {doc['num_chunks']} chunks"
                    )
            else:
                st.caption("No documents yet.")


# ─── PAGE: Integrity Analysis ─────────────────────────────────────────────────
elif page == "🔍 Integrity Analysis":
    st.title("🔍 Research Integrity Analysis")
    st.caption(
        "Automatically screens papers for p-hacking, HARKing, selective reporting, "
        "Benford's Law violations, multiple comparison issues, and data transparency problems."
    )

    collections = get_collections()
    if not collections:
        st.warning("No collections found. Upload papers first.")
        st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        sel_col = st.selectbox("📂 Collection", collections, key="int_col")

    with col2:
        docs = api_get(f"/analysis/documents/{sel_col}") or []
        if not docs:
            st.warning("No documents in this collection.")
            st.stop()
        doc_names = [d["filename"] for d in docs]
        sel_doc = st.selectbox("📄 Document to Analyse", doc_names)

    st.info(
        "💡 **What this checks:**\n"
        "- 📣 **P-value clustering** just below 0.05 (p-hacking signal)\n"
        "- 📊 **High significance rate** — all results significant (selective reporting)\n"
        "- 🧪 **Multiple comparisons** without correction\n"
        "- 📉 **Small sample sizes** with strong claims\n"
        "- 🔢 **Benford's Law** deviation (fabricated data signal)\n"
        "- 🗣️ **Vague significance language** (trend toward significance, etc.)\n"
        "- 🤔 **HARKing** — post-hoc hypotheses framed as a priori (AI-detected)\n"
        "- 📝 **Selective outcome reporting** — outcomes in Methods missing from Results\n"
        "- 📂 **Data transparency** — missing data, exclusions, availability"
    )

    if st.button("🔍 Run Integrity Analysis", type="primary"):
        with st.spinner("🤖 Running analysis (rule-based + AI checks)... ~30 seconds..."):
            result = api_post("/analysis/integrity", json_data={
                "collection": sel_col,
                "filename": sel_doc,
            })

        if result and "error" not in result:
            risk = result.get("overall_risk", "UNKNOWN")
            risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "CLEAN": "✅"}.get(risk, "⚪")
            score = result.get("risk_score", 0)

            st.markdown(f"## {risk_emoji} Overall Risk: `{risk}`  (score: {score})")
            st.markdown(
                f"**Document:** {result.get('filename')}  |  **Pages:** {result.get('num_pages')}"
            )

            sev = result.get("severity_counts", {})
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 HIGH flags", sev.get("HIGH", 0))
            c2.metric("🟡 MEDIUM flags", sev.get("MEDIUM", 0))
            c3.metric("🟢 LOW flags", sev.get("LOW", 0))

            st.markdown("---")

            flags = result.get("flags", [])
            if not flags:
                st.success("✅ No integrity issues detected. The paper appears methodologically sound.")
            else:
                st.markdown(f"### 🚩 {len(flags)} Flag(s) Found")

                sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
                flags_sorted = sorted(flags, key=lambda f: sev_order.get(f.get("severity", "LOW"), 2))

                for flag in flags_sorted:
                    sev_flag = flag.get("severity", "LOW")
                    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(sev_flag, "⚪")

                    with st.expander(f"{icon} [{sev_flag}] {flag.get('title', '')}"):
                        st.markdown(f"**Type:** `{flag.get('type', '')}`")
                        st.markdown(f"**Description:** {flag.get('description', '')}")

                        evidence = flag.get("evidence", [])
                        if evidence and isinstance(evidence, list):
                            st.markdown("**Evidence from paper:**")
                            for e in evidence:
                                if isinstance(e, str) and e.strip():
                                    st.markdown(f"> *{e.strip()}*")

                        vals = flag.get("values_found")
                        if vals is not None:
                            st.markdown(f"**Extracted values:** `{vals}`")

            st.markdown("---")
            st.markdown("### 📊 Extracted Statistics")
            stats = result.get("statistics", {})
            s1, s2 = st.columns(2)
            s1.metric("P-values found", stats.get("p_values_found", 0))
            s2.metric("Sample sizes found", len(stats.get("sample_sizes_found", [])))

            pvals = stats.get("p_values", [])
            if pvals:
                st.markdown("**P-values detected:**")
                st.dataframe(
                    [{"p-value": p["value"], "context": p["context"]} for p in pvals],
                    use_container_width=True,
                )

            ns = stats.get("sample_sizes_found", [])
            if ns:
                st.markdown(f"**Sample sizes detected:** {ns}")

            with st.expander("💾 Raw Analysis JSON"):
                st.json(result)

        elif result and "error" in result:
            st.error(f"❌ Analysis failed: {result['error']}")


# ─── PAGE: Evaluation ─────────────────────────────────────────────────────────
elif page == "📊 Evaluation":
    st.title("📊 RAG Evaluation Dashboard")

    st.info(
        "Run evaluation against a labeled dataset "
        "(JSON with question + expected_answer + relevant_document + relevant_page)."
    )

    collections = get_collections()
    if not collections:
        st.warning("No collections available.")
        st.stop()

    eval_collection = st.selectbox("Evaluate Collection", collections)
    dataset_path = st.text_input("Dataset Path", value="evaluation/questions.json")

    if st.button("▶️ Run Evaluation", type="primary"):
        with st.spinner("Running evaluation — this may take a few minutes..."):
            result = api_post("/evaluation/run", json_data={
                "collection": eval_collection,
                "dataset_path": dataset_path,
            })

        if result:
            st.success("✅ Evaluation complete!")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Recall@5", f"{result['recall_at_5']:.2f}")
            m2.metric("MRR", f"{result['mrr']:.2f}")
            m3.metric("Questions", result["num_questions"])
            m4.metric("Avg Retrieval", f"{result['avg_retrieval_latency_ms']:.0f} ms")
            m5.metric("Avg Total", f"{result['avg_total_latency_ms']:.0f} ms")
            st.markdown("#### Raw Results")
            st.json(result)

    with st.expander("📝 Expected Dataset Format"):
        st.json([{
            "question": "What was the sample size?",
            "expected_answer": "36 patients in group A, 35 in group B.",
            "relevant_document": "paper.pdf",
            "relevant_page": 4,
        }])
