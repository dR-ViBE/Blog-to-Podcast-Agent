import json
from langchain_tavily import TavilyCrawl
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

url = "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/"

tavily_crawl = TavilyCrawl()

# 1. Capture the response
response = tavily_crawl.invoke({"url": url, "max_depth": 2, "max_breadth": 4})

# 2. Parse the JSON string if necessary
if isinstance(response, str):
    try:
        response = json.loads(response)
    except json.JSONDecodeError:
        print("Error: The response is a string but not valid JSON.")
        print("Raw response:", response)
        response = {}

# 3. Extract the list of results
# The keys inside the 'results' list are usually 'url', 'content', 'raw_content', etc.
pages = response.get("results", [])

documents = []
for page in pages:
    # content could be None, so we force it to string with ( ... or "")
    content = page.get("content") or page.get("raw_content") or ""
    
    # Optional: Skip pages with no content to avoid ingestion issues
    if not content:
        continue

    documents.append(
        Document(
            page_content=content,
            metadata={
                "source": page.get("url"),
                "title": page.get("title", ""),
            },
        )
    )

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, chunk_overlap=50)

chunked_docs = text_splitter.split_documents(documents)

# Check if we actually have documents to ingest
if chunked_docs:
    vectorstore = Chroma.from_documents(
        documents=chunked_docs, 
        collection_name="blog_podcast_agent",
        embedding=OllamaEmbeddings(model="nomic-embed-text"),
        persist_directory="./.chroma"
    )
    print(f" Ingestion complete. Stored {len(chunked_docs)} chunks.")
else:
    print("No documents were found or chunked. Please check the crawl results.")