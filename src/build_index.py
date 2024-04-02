import os
from dotenv import load_dotenv
from pinecone import Pinecone
import cohere



load_dotenv()
# Load the API keys from the .env file, and initialize the clients
cohere_api_key = os.getenv('COHERE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')


co = cohere.Client(cohere_api_key)
pc = Pinecone(api_key=pinecone_api_key)
# already existing index on my pinecone account. Create a new, or use an existing one you already have. 
index = pc.Index("war-and-peace-index")


# Retrieve the book data, can be any book in txt format
def retrieve_book_data():
    with open('war_and_peace.txt', 'r') as file:
        data = file.read()
        chunk_length = 1000
        chunks = [] 
        for i in range (0, len(data), chunk_length):
            chunks.append(data[i:i+chunk_length])
        return chunks




if __name__ == '__main__':

    # Retrieve the book data and split it into chunks
    chunks = retrieve_book_data()
    batch_length = 96
    print(len(chunks))
    batches = [chunks[i:i + batch_length] for i in range(0, len(chunks), batch_length)]

    id = 0

    # Embed the chunks and upsert them into the Pinecone index
    batch = batches[0]
    for batch_number,batch in enumerate(batches):
        print(f'Batch {batch_number} of {len(batches)}')
        embeds = co.embed(
            texts=batch,
            model='embed-multilingual-v3.0',
            input_type='search_document',
            truncate='END'
        ).embeddings
        for_upsert = []
        for i, chunk in enumerate(batch):
            temp = {
                'id': str(id),
                'values': embeds[i],
                'metadata': {'text': chunk}
            }
            for_upsert.append(temp)
            id += 1
        index.upsert(vectors=for_upsert)
        print(index.describe_index_stats())

    
    
        
    

