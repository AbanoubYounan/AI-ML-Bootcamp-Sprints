import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Load the model and tokenizer
MODEL_PATH = "./saved_model"

@st.cache_resource
def load_model():
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer

model, tokenizer = load_model()

# Load the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

st.title("Question Answering with Transformers")

# Text input for context
context = st.text_area("Enter the context:", "Type or paste the context here...")

# Text input for question
question = st.text_input("Enter your question:", "Type your question here...")

# Run inference when the user clicks the button
if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.subheader("Answer:")
        st.write(result["answer"])
        st.write(f"Confidence: {result['score']:.4f}")
    else:
        st.warning("Please enter both a question and context.")
