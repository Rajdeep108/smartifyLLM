
# SmartifyLLM

SmartifyLLM is a Python package designed to make **local LLMs (Large Language Models) web-aware, context-aware, and dynamically updated without requiring model retraining, fine-tuning, or solely relying on RAG**. It enables local LLMs to fetch, process, and rank online information in real time, making responses more relevant and up-to-date.

## Features

-   **Web Awareness:** Automatically fetches and processes relevant web content, ensuring responses from local LLM are accurate, well-informed, and up-to-date. (Making local LLM smarter)
    
-   **In-house RAG system:** Implements an in-house, fast, and lightweight RAG (Retrieval Augmented Generation) approach for enhanced efficiency.
    
-   **Customizable Prompts:** Supports both predefined and user-defined prompts tailored to specific models and tasks.
    
-   **Optimized Text Splitter:** Supports PDF, DOCX, and plain text processing.

-   **Stock Market Data**: Get real-time stock prices. **(NEW UPDATE 🚩)**
    
-   **Smart Search Ranking:** Uses semantic similarity to rank sources.
    
-   **Efficient Caching:** Avoids redundant downloads and improves performance.
    

## Installation

```sh
pip install smartifyLLM
```
## Quick Start Guide 🚀

### 1. Smartify: Making LLMs Web & Context Aware

```python
from smartifyLLM import Smartify
from transformers import pipeline

# Load an LLM pipeline (Example using Phi-2)
llm = pipeline("text-generation", model="microsoft/phi-2")

# Initialize Smartify
smart_llm = Smartify(llm)

# Setting up prompt specific for phi2 model - NOTE ! {context} and {query) must be present in the prompt template. Refer to point 5 for more info.
question =  "What is the real name of ishowspeed?"
prompt_temp = """Instruct: {query) Answer based on this context: {context}
                 Output:		
              """
# Get an intelligent response
response = smart_llm.smart_response(query=question,custom_prompt=prompt_temp)
print(response)
```

#### Output Example:

Without Smartify (Phi-2):
```
"The real name of ishowspeed is Ishowspeed."
```

With Smartify:
```
"The real name of ishowspeed is Darren Jason Watkin Jr., source: Wikipedia"
```

**`Smartify.smart_response` Parameters:**

-   `query` (str): The user query.
    
-   `custom_prompt` (str, optional): A custom prompt format. **Important:** The custom prompt must include both `{context}` and `{query}` , exactly as shown (case-sensitive) for proper functionality. Any deviation will result in errors.
    
-   `custom_context` (dict, optional): A dictionary containing predefined context to get context aware answers from model.
    
-   `max_context_tokens` (int, default=4000): Max token limit for context passed into LLM.
    
-   `buffer` (int, default=200): Buffer space to avoid overloading model.
    
-   `return_source` (bool, default=False): Whether to return source URLs along with answer.
-   `advanced_stockmrkt` **- New Parameter 🚩**

| Parameter            | Type   | Default | Description                                                                 |
|----------------------|--------|---------|-----------------------------------------------------------------------------|
| `advanced_stockmrkt` | `bool` | `False` | If `True`, detects stock-related queries and returns real-time price data. |

#### Example usage:

```python
question = "What is the price of Google stock today?"
response = smart_llm.smart_response(query=question, advanced_stockmrkt=True)
print(response)
```
##### Output:

```
The current price of Google stock (GOOG) is $147.74 (Source: Yahoo Finance).
```

### 2. Text Splitting (Handling Large Documents)

```python
from smartifyLLM import TextSplitter

splitter = TextSplitter(chunk_size=300, chunk_overlap=50)
text_chunks = splitter.split_text("This is a long document that needs splitting...")
print(text_chunks)

```

**`TextSplitter` Parameters:**

-   `chunk_size` (int, default=300): The number of words per chunk.
    
-   `chunk_overlap` (int, default=50): Overlap size between chunks.
    

Supports:

-   `split_text(text: str) -> list[str]`
    
-   `split_pdf(pdf_path: str) -> list[str]`
    
-   `split_docx(docx_path: str) -> list[str]`
    

### 3. Retrieval-Augmented Generation (RAG) with RAGnarok

```python
from smartifyLLM import RAGnarok, TextSplitter

rag_model = RAGnarok()
splitter = TextSplitter()

# Split text before using vector database
document_text = "AI is transforming the world. Machine learning enables AI."
text_chunks = splitter.split_text(document_text)

rag_model.vectordb_from_document(text_chunks)
retrieved_docs = rag_model.retrieve("How is AI transforming society?", top_k=2)
print(retrieved_docs)

```

**`RAGnarok` Methods:**

-   `vectordb_from_document(chunks: List[str])`: Creates an embedding database.
    
-   `saveDB_to_disk(path: str)`: Saves the vector database to disk.
    
-   `loadDB_from_disk(path: str)`: Loads a vector database from disk.
    
-   `retrieve(query: str, top_k: int) -> List[Dict]`: Finds the most relevant documents.
    
-   `generate(model_pipeline, query: str, **kwargs) -> str`: Generates an answer using RAG.

### 4. Web Search & Retrieval

```python
from smartifyLLM import get_online_results

# Fetch web content related to a query
web_data = get_online_results("Latest advancements in AI", num_results=3)
print(web_data)

```

**`get_online_results` Parameters:**

-   `query` (str): The search query.
    
-   `num_results` (int, default=5): Number of search results to show.
    
-   `paragraphs` (int, default=5): Number of paragraphs to extract per page.
    
-   `min_delay` (int, default=2): Min delay between requests.
    
-   `max_delay` (int, default=5): Max delay between requests.
    
-   `verify_ssl` (bool, default=True): Whether to verify SSL certificates of websites.
    
-   `region` (str, optional): Country-specific search filtering.
    

### 5. Custom Prompting

```python
from smartifyLLM import DEFAULT_PROMPT

#or you can make a custom prompt of your own ; must have "{query}" and "{context}" inside the prompt
custom_prompt = """
Give me a summarized, Shakespearean answer based on this context:
{context}

Question: {query}
Answer:
"""

```
**Note:** The placeholders `{context}` and `{query}` must be present in the custom prompt with the exact same spelling and case, or else an error will occur. The `smart_response` method expects `{context}` and `{query}` to be provided indefinitely by the user in the custom prompt for proper functioning.

# Functions & Methods (Detailed guide)

Here's a detailed breakdown of all available functions and methods in each module.

## 1. RAGnarok (Retrieval Augmented Generation)

### `from smartifyLLM import RAGnarok`

Provides tools for creating vector embedding from documents, retrieving relevant information, and generating responses.

- `vectordb_from_document(chunks: List[str]) -> None`
  - Converts a list of text chunks into a searchable vector database.

- `_update_vectordb(texts: List[str]) -> None`
  - Updates the vector database with new embeddings.

- `saveDB_to_disk(path: str) -> None`
  - Saves the vector database to disk as a `.pt` file.
  - **Parameters:** `path` (must end with `.pt`)

- `loadDB_from_disk(path: str) -> None`
  - Loads a previously saved vector database from disk.
  - **Parameters:** `path` (must be an existing `.pt` file)

- `retrieve(query: str, top_k: int = 3) -> List[Dict]`
  - Retrieves the most relevant documents based on the input query.
  - **Parameters:** `query` (search term), `top_k` (number of results to return)
  - **Returns:** List of dictionaries containing content, similarity score, and ranking.

- `generate(model_pipeline, query: str, **kwargs) -> str`
  - Generates a response based on retrieved documents.
  - **Parameters:** `model_pipeline` (language model function), `query` (input query)
  - **Returns:** A generated response.
 

## 2. Smartify

### `from smartifyLLM import Smartify`

Enhances responses by incorporating online search results and structured context.

- `Base argument: (model_pipeline)`
    - **args:**
         - Accepts the language model pipeline to be made smart.

- `smart_response(query: str, custom_prompt: str = None, custom_context: Optional[Dict[str, str]] = None, max_context_tokens: int = 4000, buffer: int = 200, return_source: bool = False) -> Union[str, Tuple[str, List[str]]]`
  - Generates a smart response by retrieving the best available context (either online or from a given custom context).
  - **Parameters:**
    - `query`: The input question.
    - `custom_prompt`: Custom formatting for the response.
    - `custom_context`: A dictionary of possible sources.
    - `max_context_tokens`: Maximum tokens for context (default: 4000).
    - `buffer`: Safety buffer for token overflow (default: 200).
    - `return_source`: If `True`, returns the source alongside the response.
  - **Returns:** The generated response or a tuple (`response`, `sources`).


## 3. TextSplitter (Document Processing & Chunking)

### `from smartifyLLM import TextSplitter`

This module splits text documents into manageable chunks for better retrieval and processing.

- `Base argument: (chunk_size: int = 300, chunk_overlap: int = 50)`
  - **args:**
    - `chunk_size`: The number of words per chunk.
    - `chunk_overlap`: The number of overlapping words between chunks.

- `split_text(text: str) -> List[str]`
  - Splits a given raw text into chunks.
  - **Parameters:** `text` (string input to split)
  - **Returns:** A list of text chunks.

- `split_pdf(pdf_path: str) -> List[str]`
  - Extracts text from a PDF file and splits it into chunks.
  - **Parameters:** `pdf_path` (path to a `.pdf` file)
  - **Returns:** A list of text chunks.

- `split_docx(docx_path: str) -> List[str]`
  - Extracts text from a Word document and splits it into chunks.
  - **Parameters:** `docx_path` (path to a `.docx` file)
  - **Returns:** A list of text chunks.


## 4. Web Tools (Search & Scraping)

### Functions

These functions facilitate web searches, content scraping, and result ranking.

- `get_random_headers() -> Dict[str, str]`
  - Returns randomized headers for web requests to reduce detection.

- `search_urls(query: str, num_results: int = 5, region: Optional[str] = None) -> List[str]`
  - Fetches a list of relevant URLs based on a search query.
  - **Parameters:**
    - `query`: Search term.
    - `num_results`: Number of URLs to retrieve.
    - `region`: Optional region for search.
  - **Returns:** A list of URLs.

- `scrape_website(url: str, paragraphs: int = 5, min_delay: int = 2, max_delay: int = 5, verify_ssl: bool = True) -> str`
  - Scrapes a webpage for textual content.
  - **Parameters:**
    - `url`: Website URL.
    - `paragraphs`: Number of paragraphs to extract.
    - `min_delay`, `max_delay`: Randomized delays to prevent blocking.
    - `verify_ssl`: Whether to verify SSL certificates.
  - **Returns:** Extracted content as a string.

- `get_online_results(query: str, num_results: int = 5, paragraphs: int = 5, min_delay: int = 2, max_delay: int = 5, verify_ssl: bool = True, region: Optional[str] = None) -> Dict[str, str]`
  - Performs an online search and retrieves content from relevant sources.
  - **Returns:** A dictionary mapping URLs to extracted text.

- `rank_best_answer(query: str, answers_dict: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]`
  - Identifies the most relevant answer from a dictionary of responses.
  - **Returns:** A tuple containing the best source and content.



## 5. Utility Functions

These functions provide additional support for managing models and resources.

- `clear_model_cache() -> None`
  - Clears all cached models to free up memory.

## 6. Real-Time Stock Market Info (NEW FEATURE 🚨)

SmartifyLLM now includes built-in support for **real-time stock price retrieval**.  
Just set `advanced_stockmrkt=True`(default = False) in `smart_response`.


#### Use Case:

```python
from smartifyLLM import Smartify
from transformers import pipeline

llm = pipeline("text-generation", model="gpt2")  # or any local model
smart_llm = Smartify(llm)

question = "What is the price of Google stock today?"
response = smart_llm.smart_response(query=question, advanced_stockmrkt=True)
print(response)
```

---

### Output Example:

```
The current price of Google stock (GOOG) is $147.74 (Source: Yahoo Finance).
```
### Output Without Smartify: (Standard Local LLM Output)

Local LLMs without access to real-time data may hallucinate or return outdated/incorrect values:

```
"What is the price of tesla stock today?"
```
**LLM output:**
```
"The price of tesla stock today is $1,895.00."
```

With Smartify and advanced_stockmrkt=True, you get real-time, accurate results.

## License

SmartifyLLM is licensed under the **MIT License**.