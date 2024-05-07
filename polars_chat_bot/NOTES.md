# Setup/Install

1. Scrape all the documentation from the site: https://docs.pola.rs/py-polars/html/reference/: `wget --convert-links -r -l 2 'https://docs.pola.rs/py-polars/html/reference/index.html'`
3. Install ollama: https://ollama.com/download
4. Install llama3: `ollama pull llama3`
5. Setup the model in Pinecone; need to match dimensions of embedding to whatever the library produces; 4096 in this case
6. 