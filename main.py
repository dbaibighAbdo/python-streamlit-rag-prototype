import streamlit as st
from helpers.fileProccessing import process_pdf
from helpers.ragService import langchain_expression_language
from dotenv import load_dotenv

def main():
    load_dotenv()
    st.title("RAG Prototype")
    st.set_page_config(page_title="RAG Prototype", layout="wide")
    st.header("Welcome to the RAG Prototype Application")

    pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf is not None:
        st.success("PDF uploaded successfully!")
        # Further processing can be done here
        knowledge_base = process_pdf(pdf)

        userQuery = st.text_input("Enter your question:")
        if userQuery:
            with st.spinner("Generating answer..."):
                answer = langchain_expression_language(userQuery, knowledge_base)
            st.subheader("Answer:")
            st.write(answer)


if __name__ == "__main__":
    main()
