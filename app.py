import streamlit as st
import pandas as pd

from src.qa import PlantDocAssistant

st.set_page_config(page_title="Plant Documentation Router", layout="wide")

st.title("Plant Documentation Router")
st.caption("Ask a plant-floor question and the system will route it to the right documentation source.")

@st.cache_resource
def load_assistant():
    return PlantDocAssistant()

assistant = load_assistant()

question = st.text_area(
    "Ask a plant-floor question",
    placeholder="Example: What should happen if two consecutive samples fail dimensional inspection?"
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Routing and retrieving answer..."):
            result = assistant.answer(question)

        st.subheader("Question Routed To")
        st.success(result["source_label"])

        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("grounded") == "no" and result.get("abstain_reason"):
            st.info(f"Reason: {result['abstain_reason']}")

        st.subheader("Routing Confidence")
        route_df = pd.DataFrame(
            [{"source": k, "score": v} for k, v in result["route_scores"].items()]
        ).sort_values("score", ascending=False)
        st.dataframe(route_df, use_container_width=True)

        st.subheader("Source Files Used")
        source_files = list({chunk["file_name"] for chunk in result["retrieved_chunks"]})
        for file_name in source_files:
            st.write(f"- {file_name}")

        with st.expander("Retrieved Chunks"):
            for i, chunk in enumerate(result["retrieved_chunks"], start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(f"Heading: {chunk.get('heading', 'General')}")
                st.write(f"File: {chunk['file_name']}")
                st.write(f"Hybrid Score: {chunk.get('hybrid_score', 0):.4f}")
                st.write(f"Dense Score: {chunk.get('dense_score', 0):.4f}")
                st.write(f"Sparse Score: {chunk.get('sparse_score', 0):.4f}")
                st.write(chunk["chunk_text"])
                st.divider()