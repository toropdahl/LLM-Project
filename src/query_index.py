import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import cohere
import numpy as np
import itertools
from openai import OpenAI


def clean_up_text(text):
    # Replace single newlines with spaces
    cleaned_text = (
        text.replace("\n\n", "\x00").replace("\n", " ").replace("\x00", "\n\n")
    )
    return cleaned_text


load_dotenv()
cohere_api_key = os.getenv("COHERE_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")


co = cohere.Client(cohere_api_key)
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("war-and-peace-index")
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


queries = [
    "Who is Pierre Bezukhov and what is his significance in 'War and Peace'?",
    "Describe the battle of Borodino and its significance in the narrative of 'War and Peace'.",
    "How does the character of Natasha Rostova evolve from the beginning to the end of 'War and Peace'?",
    "What is the philosophical debate between Pierre Bezukhov and Prince Andrei Bolkonsky about life and death in 'War and Peace'?",
    "In 'War and Peace', how does Tolstoy portray the French invasion of Russia and its impact on the Russian people?",
    "Who are the members of the Rostov family, and what roles do they play in 'War and Peace'?",
    "How does the relationship between Pierre Bezukhov and Helene Kuragina evolve throughout 'War and Peace'?",
    "What are the circumstances leading to Prince Andrei Bolkonsky's death in 'War and Peace'?",
    "How does Tolstoy illustrate the contrast between the lives of soldiers and civilians in 'War and Peace'?",
    "In 'War and Peace', what event leads to Natasha Rostova's disillusionment with Prince Anatol Kuragin?",
]
if __name__ == "__main__":

    #query = queries[1]
    query = "Are there any embarrassing moments in the book?"
    xq = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        truncate="END",
    ).embeddings

    res = index.query(vector=xq, top_k=5, include_metadata=True)

    res = res["matches"]
    text_string = ""
    for i in res:

        text_string += (f"Relevant paragraph {i.id}: " + clean_up_text(i.metadata["text"]))
        text_string += "\n"

    message = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions. Dont mention that the information is from something called context. The information that could be relevant to answer the question is provided in the CONTEXT. Answer the users query based purely on the context provided. Say if you can not answer the question",
        },
        {"role": "user", "content": "CONTEXT: " + text_string},
        {"role": "user", "content": "QUERY: " + query},
    ]
    response = client.chat.completions.create(
        messages=message,
        model="gpt-3.5-turbo",
    )
    # Extracting the content from the response
    response_content = response.choices[0].message.content
    print("QUERY: \n" + query)
    print("RESPONSE: \n" + response_content)
