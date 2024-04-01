import streamlit as st
import os
import cohere
from pinecone import Pinecone
from openai import OpenAI



# Initialize Cohere, Pinecone, and OpenAI
co = cohere.Client(st.secrets["COHERE_API_KEY"])
pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
index = pc.Index("war-and-peace-index")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Streamlit UI Setup
st.title("ChatGPT-like Clone with War and Peace")

# Initialize or retrieve session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation
for message in st.session_state.messages:
    with st.container():
        st.write(f"{message['role'].capitalize()}: {message['content']}")

# User input
user_query = st.text_input("Ask me anything about 'War and Peace':", key="user_query")

# Function to clean up text
def clean_up_text(text):
    return text.replace("\n\n", "\x00").replace("\n", " ").replace("\x00", "\n\n")

# Process query and display response
if st.button("Submit"):
    # Embed the user query with Cohere
    xq = co.embed(
        texts=[user_query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        truncate="END",
    ).embeddings    
    # Query Pinecone index
    res = index.query(vector=xq, top_k=5, include_metadata=True)
    res = res["matches"]
    context = ""
    for i in res:

        context += (f"Relevant paragraph {i.id}: " + clean_up_text(i.metadata["text"]))
        context += "\n"
    # Create message list for OpenAI Chat Completion
    message = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions. Dont mention that the information is from something called context. The information that could be relevant to answer the question is provided in the CONTEXT. Answer the users query based purely on the context provided. Say if you can not answer the question",
        },
        {"role": "user", "content": "CONTEXT: " + context},
        {"role": "user", "content": "QUERY: " + user_query},
    ]

    # Send context + query to OpenAI
    response = client.chat.completions.create(
        messages=message,
        model="gpt-4-0125-preview",
    )

    # Extracting and displaying the response
    response_content = response.choices[0].message.content
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "assistant", "content": response_content})
    st.experimental_rerun()
