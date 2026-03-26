from pdf2image import convert_from_path
from pathlib import Path
from pytesseract import image_to_string
import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


data_dir = Path('data')
report = data_dir / 'Отчет Росатом 2018.pdf'
collection_name = 'rosatom'
embed_model = 'yxchia/multilingual-e5-base:latest'
llm_model = 'qwen2.5:3b'

images = convert_from_path(report)

text = ''
for i, image in enumerate(images):
    text += image_to_string(image, lang='rus+eng')



def make_chunks(text):
    custom_separators = [
        r"\n\n",          # Абзацы
        r"\n",            # Строки
        r"\.\s",          # Конец предложения (точка и пробел)
        r"[0-9]\.",     # Нумерованные списки (напр. "1.")
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=custom_separators,
        is_separator_regex=True  # Ключевой параметр для активации regex
    )

    chunks = splitter.split_text(text)

    return chunks


def embedder(texts, mode='query', embed_model='yxchia/multilingual-e5-base:latest'):

    if mode not in ['query', 'passage']:
        raise
    
    texts = [f"{mode}: {t}" for t in texts]

    embeds = ollama.embed(model=embed_model, 
                        input=texts)

    return embeds['embeddings']


chunks = make_chunks(text)

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(collection_name,
                                metadata={"hnsw:space": "cosine"})

client.delete_collection(name=collection_name)
collection = client.get_or_create_collection(collection_name,
                                metadata={"hnsw:space": "cosine"})


embeddings = embedder(chunks, mode='passage')
ids = [f"chunk_{i}" for i in range(len(chunks))]

collection.add(ids, embeddings=embeddings, documents=chunks)




