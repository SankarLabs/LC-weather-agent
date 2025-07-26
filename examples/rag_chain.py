"""
RAG (Retrieval-Augmented Generation) Implementation Example

This file demonstrates a complete RAG system implementation using LangChain.
It shows best practices for document processing, vector storage, retrieval,
and question answering with source attribution.

Key Patterns Demonstrated:
- Document loading and processing
- Text splitting and chunking strategies
- Vector store setup and management
- Retrieval chain configuration
- Source attribution and metadata handling
- Performance optimization techniques
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.document_loaders import (
    PDFLoader, 
    TextLoader, 
    WebBaseLoader,
    DirectoryLoader
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from pydantic import BaseModel, Field
import tiktoken

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGConfig(BaseModel):
    """Configuration for RAG system."""
    chunk_size: int = Field(1000, description="Size of text chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    k_documents: int = Field(3, description="Number of documents to retrieve")
    temperature: float = Field(0.1, description="LLM temperature for factual responses")
    max_tokens: int = Field(500, description="Maximum tokens in response")
    persist_directory: str = Field("./chroma_db", description="Vector store persistence directory")

class RAGResponse(BaseModel):
    """Structured response from RAG system."""
    success: bool
    answer: Optional[str] = None
    sources: List[Dict[str, Any]] = []
    error: Optional[str] = None
    execution_time: Optional[float] = None
    tokens_used: Optional[int] = None

class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]  # Hierarchical splitting
        )
        
        # Initialize tokenizer for accurate counting
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Load documents from a directory with multiple file types.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List[Document]: Loaded documents with metadata
        """
        documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.warning(f"Directory {directory_path} does not exist")
            return documents
        
        # Load different file types
        file_loaders = {
            '.pdf': PDFLoader,
            '.txt': TextLoader,
            '.md': TextLoader,
        }
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in file_loaders:
                try:
                    loader_class = file_loaders[file_path.suffix.lower()]
                    loader = loader_class(str(file_path))
                    file_docs = loader.load()
                    
                    # Add file metadata
                    for doc in file_docs:
                        doc.metadata.update({
                            'file_path': str(file_path),
                            'file_name': file_path.name,
                            'file_type': file_path.suffix,
                            'file_size': file_path.stat().st_size,
                            'processed_at': time.time()
                        })
                    
                    documents.extend(file_docs)
                    logger.info(f"Loaded {len(file_docs)} documents from {file_path.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Total documents loaded: {len(documents)}")
        return documents
    
    def load_web_documents(self, urls: List[str]) -> List[Document]:
        """
        Load documents from web URLs.
        
        Args:
            urls: List of URLs to load
            
        Returns:
            List[Document]: Loaded web documents
        """
        documents = []
        
        for url in urls:
            try:
                loader = WebBaseLoader(url)
                web_docs = loader.load()
                
                # Add URL metadata
                for doc in web_docs:
                    doc.metadata.update({
                        'source_url': url,
                        'content_type': 'web',
                        'processed_at': time.time()
                    })
                
                documents.extend(web_docs)
                logger.info(f"Loaded document from {url}")
                
            except Exception as e:
                logger.error(f"Failed to load {url}: {e}")
        
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks with metadata preservation.
        
        Args:
            documents: Documents to split
            
        Returns:
            List[Document]: Split document chunks
        """
        if not documents:
            logger.warning("No documents to split")
            return []
        
        start_time = time.time()
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add chunk metadata and token counts
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'chunk_id': i,
                'chunk_size': len(doc.page_content),
                'token_count': len(self.tokenizer.encode(doc.page_content))
            })
        
        processing_time = time.time() - start_time
        avg_chunk_size = sum(len(doc.page_content) for doc in split_docs) / len(split_docs)
        
        logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
        logger.info(f"Average chunk size: {avg_chunk_size:.0f} characters")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        
        return split_docs
    
    def create_sample_documents(self) -> List[Document]:
        """Create sample documents for testing when no files are available."""
        sample_docs = [
            Document(
                page_content="""
                LangChain is a framework for developing applications powered by language models.
                It enables applications that are context-aware and can reason about their environment.
                The main value props of LangChain are:
                1. Components: modular abstractions for the components necessary to work with language models
                2. Chains: structured assemblies of components for specific tasks
                3. Agents: systems that use language models to determine which actions to take
                """,
                metadata={"source": "sample_langchain_intro", "type": "educational"}
            ),
            Document(
                page_content="""
                RAG (Retrieval-Augmented Generation) is a technique that combines the power of 
                large language models with external knowledge retrieval. It works by:
                1. Indexing external documents in a vector database
                2. Retrieving relevant documents based on user queries
                3. Using the retrieved context to generate more accurate responses
                RAG helps reduce hallucinations and provides source attribution.
                """,
                metadata={"source": "sample_rag_explanation", "type": "educational"}
            ),
            Document(
                page_content="""
                Vector databases are specialized databases designed to store and query vector embeddings.
                Popular vector databases include:
                - Chroma: Open-source, easy to use, good for development
                - Pinecone: Managed service, highly scalable
                - Weaviate: Open-source with strong semantic search capabilities
                - FAISS: Facebook's library for efficient similarity search
                Vector databases enable semantic search and are essential for RAG systems.
                """,
                metadata={"source": "sample_vector_db_info", "type": "educational"}
            )
        ]
        
        logger.info("Created sample documents for demonstration")
        return sample_docs

class RAGSystem:
    """Complete RAG system implementation."""
    
    def __init__(self, config: RAGConfig = RAGConfig()):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.embeddings = None
        self.vectorstore = None
        self.retrieval_chain = None
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize embeddings and LLM components."""
        try:
            # Initialize embeddings
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Test embeddings
            test_embedding = self.embeddings.embed_query("test")
            logger.info(f"Embeddings initialized successfully (dimension: {len(test_embedding)})")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            logger.info("RAG system components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        Create vector store from documents.
        
        Args:
            documents: Documents to index
            
        Returns:
            Chroma: Configured vector store
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")
        
        start_time = time.time()
        
        try:
            # Create vector store
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.config.persist_directory,
                collection_metadata={"hnsw:space": "cosine"}
            )
            
            # Persist the store
            vectorstore.persist()
            
            creation_time = time.time() - start_time
            logger.info(f"Vector store created with {len(documents)} documents in {creation_time:.2f}s")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise
    
    def load_existing_vector_store(self) -> Optional[Chroma]:
        """
        Load existing vector store from disk.
        
        Returns:
            Optional[Chroma]: Loaded vector store or None if not found
        """
        if not Path(self.config.persist_directory).exists():
            logger.info("No existing vector store found")
            return None
        
        try:
            vectorstore = Chroma(
                persist_directory=self.config.persist_directory,
                embedding_function=self.embeddings
            )
            
            # Test the store
            test_results = vectorstore.similarity_search("test", k=1)
            logger.info(f"Loaded existing vector store with {len(test_results)} test results")
            
            return vectorstore
            
        except Exception as e:
            logger.error(f"Failed to load existing vector store: {e}")
            return None
    
    def setup_retrieval_chain(self, vectorstore: Chroma) -> RetrievalQA:
        """
        Set up the retrieval QA chain.
        
        Args:
            vectorstore: Vector store for retrieval
            
        Returns:
            RetrievalQA: Configured retrieval chain
        """
        try:
            # Create retriever with multiple search strategies
            retriever = vectorstore.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={
                    "k": self.config.k_documents,
                    "lambda_mult": 0.7  # Balance relevance vs diversity
                }
            )
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",  # Stuff all context into prompt
                retriever=retriever,
                return_source_documents=True,
                verbose=True,
                chain_type_kwargs={
                    "prompt": self._create_qa_prompt()
                }
            )
            
            logger.info("Retrieval chain configured successfully")
            return qa_chain
            
        except Exception as e:
            logger.error(f"Failed to setup retrieval chain: {e}")
            raise
    
    def _create_qa_prompt(self):
        """Create custom QA prompt with source attribution."""
        from langchain.prompts import PromptTemplate
        
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        Always cite the sources you used to answer the question.

        Context:
        {context}

        Question: {question}
        
        Answer with source attribution:"""
        
        return PromptTemplate(template=template, input_variables=["context", "question"])
    
    def initialize_from_directory(self, directory_path: str) -> bool:
        """
        Initialize RAG system from a directory of documents.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Try to load existing vector store first
            self.vectorstore = self.load_existing_vector_store()
            
            if self.vectorstore is None:
                # Load and process documents
                logger.info(f"Loading documents from {directory_path}")
                documents = self.processor.load_documents_from_directory(directory_path)
                
                if not documents:
                    logger.warning("No documents found, using sample documents")
                    documents = self.processor.create_sample_documents()
                
                # Split documents
                split_docs = self.processor.split_documents(documents)
                
                # Create vector store
                self.vectorstore = self.create_vector_store(split_docs)
            
            # Set up retrieval chain
            self.retrieval_chain = self.setup_retrieval_chain(self.vectorstore)
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def query(self, question: str) -> RAGResponse:
        """
        Query the RAG system with source attribution.
        
        Args:
            question: User question
            
        Returns:
            RAGResponse: Response with answer and sources
        """
        if not self.retrieval_chain:
            return RAGResponse(
                success=False,
                error="RAG system not initialized. Call initialize_from_directory() first."
            )
        
        start_time = time.time()
        
        try:
            logger.info(f"Processing query: {question[:100]}...")
            
            # Execute the retrieval chain
            result = self.retrieval_chain({"query": question})
            
            # Extract source information
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    sources.append(source_info)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Query processed successfully in {execution_time:.2f}s")
            
            return RAGResponse(
                success=True,
                answer=result.get("result", ""),
                sources=sources,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Query processing failed: {str(e)}"
            logger.error(error_msg)
            
            return RAGResponse(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to existing vector store.
        
        Args:
            documents: New documents to add
            
        Returns:
            bool: True if successful
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return False
        
        try:
            # Split new documents
            split_docs = self.processor.split_documents(documents)
            
            # Add to vector store
            self.vectorstore.add_documents(split_docs)
            self.vectorstore.persist()
            
            logger.info(f"Added {len(split_docs)} document chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def get_similar_documents(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Get similar documents with similarity scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List[Tuple[Document, float]]: Documents with similarity scores
        """
        if not self.vectorstore:
            logger.error("Vector store not initialized")
            return []
        
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

def demonstrate_rag_system():
    """Demonstrate the RAG system with examples."""
    
    # Validate environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        return
    
    print("\n" + "="*60)
    print("RAG SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Initialize RAG system
    config = RAGConfig(
        chunk_size=800,
        chunk_overlap=150,
        k_documents=3,
        temperature=0.1
    )
    
    rag_system = RAGSystem(config)
    
    # Initialize with sample documents (or from directory if available)
    documents_dir = "documents"  # Change this to your documents directory
    
    print(f"\nInitializing RAG system from {documents_dir}...")
    if not rag_system.initialize_from_directory(documents_dir):
        logger.error("Failed to initialize RAG system")
        return
    
    # Example queries
    test_queries = [
        "What is LangChain and what are its main components?",
        "How does RAG work?",
        "What are the different types of vector databases?",
        "What are the benefits of using retrieval-augmented generation?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n" + "-"*50)
        print(f"QUERY {i}: {query}")
        print("-"*50)
        
        response = rag_system.query(query)
        
        if response.success:
            print(f"ANSWER: {response.answer}")
            print(f"\nSOURCES ({len(response.sources)}):")
            for j, source in enumerate(response.sources, 1):
                print(f"  {j}. {source['content']}")
                print(f"     Metadata: {source['metadata']}")
            print(f"\nExecution time: {response.execution_time:.2f}s")
        else:
            print(f"ERROR: {response.error}")
    
    # Demonstrate similarity search
    print(f"\n" + "-"*50)
    print("SIMILARITY SEARCH EXAMPLE")
    print("-"*50)
    
    similar_docs = rag_system.get_similar_documents("vector databases", k=3)
    
    for i, (doc, score) in enumerate(similar_docs, 1):
        print(f"\nDocument {i} (Score: {score:.3f}):")
        print(f"Content: {doc.page_content[:150]}...")
        print(f"Metadata: {doc.metadata}")

def create_test_documents():
    """Create test documents for demonstration."""
    documents_dir = Path("documents")
    documents_dir.mkdir(exist_ok=True)
    
    # Create sample documents
    sample_files = {
        "langchain_intro.txt": """
        LangChain is a framework for developing applications powered by language models.
        It enables applications that are context-aware and can reason about their environment.
        
        The main components of LangChain include:
        
        1. LLMs and Chat Models: Wrappers around language models
        2. Prompts: Templates and utilities for prompts
        3. Chains: Sequences of calls to LLMs or other utilities
        4. Agents: Systems that use LLMs to determine actions
        5. Memory: Utilities for persisting state between chain calls
        6. Document Loaders: Utilities for loading documents from various sources
        
        LangChain provides a standard interface for chains, lots of integrations with other tools,
        and end-to-end chains for common applications.
        """,
        
        "rag_explanation.txt": """
        Retrieval-Augmented Generation (RAG) is a powerful technique that combines
        the capabilities of large language models with external knowledge retrieval.
        
        The RAG process works as follows:
        
        1. Document Ingestion: External documents are processed and split into chunks
        2. Embedding Creation: Each chunk is converted to a vector embedding
        3. Vector Storage: Embeddings are stored in a vector database
        4. Query Processing: User queries are converted to embeddings
        5. Similarity Search: The most relevant document chunks are retrieved
        6. Response Generation: The LLM generates responses using retrieved context
        
        Benefits of RAG:
        - Reduces hallucinations by grounding responses in real data
        - Enables access to up-to-date information
        - Provides source attribution and transparency
        - Allows for domain-specific knowledge without retraining models
        
        RAG is particularly useful for question-answering systems, chatbots,
        and knowledge management applications.
        """,
        
        "vector_databases.txt": """
        Vector databases are specialized databases designed to store, index, and query
        high-dimensional vector embeddings efficiently.
        
        Popular vector database options include:
        
        1. Chroma: Open-source, Python-native, easy to use for development
        2. Pinecone: Managed vector database service, highly scalable
        3. Weaviate: Open-source with strong semantic search capabilities
        4. Qdrant: Open-source vector search engine with advanced filtering
        5. FAISS: Facebook's library for efficient similarity search
        6. Milvus: Open-source vector database for scalable similarity search
        
        Key features to consider:
        - Scalability and performance
        - Query flexibility and filtering capabilities
        - Integration with ML frameworks
        - Deployment options (cloud vs on-premise)
        - Cost and licensing
        
        Vector databases enable semantic search, recommendation systems,
        and are essential components of RAG architectures.
        """
    }
    
    for filename, content in sample_files.items():
        file_path = documents_dir / filename
        if not file_path.exists():
            file_path.write_text(content.strip())
            logger.info(f"Created sample document: {filename}")

if __name__ == "__main__":
    """
    Run the RAG system demonstration.
    
    This demonstrates:
    1. RAG system initialization
    2. Document processing and vector store creation
    3. Query processing with source attribution
    4. Similarity search capabilities
    5. Error handling throughout the pipeline
    """
    
    try:
        # Create test documents if they don't exist
        create_test_documents()
        
        # Run the demonstration
        demonstrate_rag_system()
        
    except KeyboardInterrupt:
        logger.info("Demonstration interrupted by user")
        
    except Exception as e:
        logger.error(f"Unexpected error in demonstration: {e}")
        raise
    
    finally:
        logger.info("RAG system demonstration completed")