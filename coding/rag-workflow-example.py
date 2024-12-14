# https://github.com/yobix-ai/extractous
# Install this, use it to turn files into text -> embed with sentence-transformers 
# -> store embeddings in a flat FAISS index (if on windows, use conda for faiss-gpu, otherwise you can only use faiss-cpu, which is still great) or txtai.
# Ive tried every RAG implementation out there, and they are all nightmares to setup, awfully translated from chinese, require a dozen docker containers, or are just painfully slow.
# This is the simplest and fastest, trust me. Jinav3, cde-small, BGE-m3 are SOTA.

# pip install extractous langchain_text_splitter ollama faiss-cpu
# Install ollama to your computer first, and make sure you download the model you want to use with ollama.
# also, I am using faiss-gpu in a Conda environment, but that's a hassle to setup on windows, so if you dont want the headache just install faiss-cpu.


import ollama
import sentence_transformers
import faiss
import langchain_text_splitters
from extractous import Extractor

extractor = Extractor()
reader, metadata = extractor.extract_file("pdf-test.pdf")

result = ""
buffer = reader.read(4096)
while len(buffer) > 0:
    result += buffer.decode("utf-8")
    buffer = reader.read(4096)

print(result)

embedder = sentence_transformers.SentenceTransformer('jinaai/jina-embeddings-v3', trust_remote_code=True)

text_splitter = langchain_text_splitters.RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_text(result)

index = faiss.IndexFlatL2(1024)
embedding = embedder.encode(chunks)
index.add(embedding)

query = input()

query_embedding = embedder.encode([query]) # as a list, because encode expects multiple sentences
# Return k many results
k = 3
D, I = index.search(query_embedding, k)
context_chunks = I[0]

query = input("Enter your query: ")

relevant_chunks = []
print("\nTop 3 most relevant chunks:")
for index in I[0]:  
    relevant_chunk = chunks[index]
    relevant_chunks += relevant_chunk
    print(f"Chunk {index}:")
    print(relevant_chunk)
    print("---")  # Separator between chunks

response = ollama.generate('qwen2.5:3b', prompt=f'Context: {relevant_chunks} User Query {query}')
print(response)  
