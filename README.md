# Vector Store

### Summary

This is a simple python implementation of a vector store that showcase the ability to use an embedding model to store documents in an high dimensional vector space, and do similarity search to retrieve the most relevant documents. Furthermore it showcase how this can then be used to do RAG with an LLM like mistral without the need of libraries like Langchain or LlamaIndex

### Description of the directory

Everything relevant to the implementation is explained and implemented inside a .ipn ... file -> this

### Installation

- Install [Ollama](https://ollama.com/) for the RAG

- Install the model we are using:

```bash
ollama pull mistral
```

#### Clone the repo and move inside it

```bash
git clone https://github.com/Jac-Zac/IR_Vector_Store_RAG && cd https://github.com/Jac-Zac/IR_Vector_Store_RA
```

##### Set up a Python virtual environment

```bash
python -m venv .env
source .env/bin/activate
```

> If you are having problems with jupyter
> `ipython kernel install --name "local-venv-kernel" --user`

#### Install dependencies

```bash
pip install -r requirements.txt
```

### Data for RAG

- https://arxiv.org/pdf/2201.02177.pdf
- https://arxiv.org/pdf/1706.03762.pdf
