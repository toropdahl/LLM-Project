# Simple LLM Using Cohere, Pinecone, and OpenAI

This project demonstrates a simple Language Learning Model (LLM) utilizing APIs from Cohere for text embedding, Pinecone for vector database services, and OpenAI for generating contextual responses. Designed to work with the text of Leo Tolstoy's "War and Peace", this implementation indexes the book's text allowing for intelligent querying and conversation within a Streamlit app. The web app is hosted [here](https://llm-project-igexxext9fgjtc32s7mzfn.streamlit.app/).


## Features

- Text preprocessing and chunking for efficient data handling.
- Use of Cohere's embedding API to convert text into vector form.
- Implementation of Pinecone for indexing and querying vector data.
- Streamlit app for user interaction, leveraging OpenAI's GPT model to generate responses based on the book's content.
- Multilingual functionality.

## Getting Started

### Prerequisites

- Python 3.x
- Accounts and API keys for Cohere, Pinecone, and OpenAI.
- Streamlit (for the interactive app).

### Installation

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project directory with your API keys:
   ```plaintext
   COHERE_API_KEY=<your_cohere_api_key>
   PINECONE_API_KEY=<your_pinecone_api_key>
   OPENAI_API_KEY=<your_openai_api_key>
   ```
4. To run the indexing script:
   ```bash
   python <name_of_the_script>.py
   ```
   Ensure you replace `<name_of_the_script>` with the actual script name that performs the indexing. You also need a book to upload to pinecone, in txt format. Other changes that need to be made is mentioned in the build_index.py file.

### Usage

To use the Streamlit app:

1. Ensure you have indexed "War and Peace" by running the indexing script as mentioned above.
2. Start the Streamlit app:
   ```bash
   streamlit run <streamlit_script_name>.py
   ```
   Replace `<streamlit_script_name>` with the name of the Streamlit app script. 

## How It Works

- **Text Preprocessing and Indexing**: The first script preprocesses "War and Peace", dividing it into manageable chunks, and uses Cohere's embedding API to convert these chunks into vectors. These vectors, along with their corresponding text chunks, are then indexed using Pinecone.
- **Streamlit App**: The app allows users to query the indexed data. It uses Cohere to embed user queries, Pinecone to find the most relevant text chunks, and OpenAI's GPT model to generate responses based on this context.


## License

This project is open source.
