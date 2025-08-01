# Advanced Retrieval-Augmented Generation (RAG) System
# This implementation includes embedding generation, document processing,
# vector retrieval, and reranking capabilities for improved query responses

# Import required libraries
from BCEmbedding.tools.llama_index import BCERerank
import os
import logging
from typing import List, Optional, Tuple, Dict
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    SimpleDirectoryReader,
    Document
)
from llama_index.node_parser import SimpleNodeParser
from llama_index.llms import Ollama
from llama_index.retrievers import VectorIndexRetriever
from llama_index.schema import NodeWithScore

# Configure logging to track system behavior and troubleshoot issues
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_system.log"),  # Log to file
        logging.StreamHandler()                 # Log to console
    ]
)
logger = logging.getLogger("AdvancedRAGSystem")

class AdvancedRAGSystem:
    """
    A comprehensive Retrieval-Augmented Generation (RAG) system that combines
    embedding models, document processing, vector retrieval, and reranking
    to provide accurate and contextually relevant responses to user queries.
    """

    def __init__(
        self,
        embedding_model_name: str = "maidalun1020/bce-embedding-base_v1",
        reranker_model_name: str = "maidalun1020/bce-reranker-base_v1",
        llm_model_name: str = "qwen2:0.5b",
        device: str = "cpu",
        max_embedding_length: int = 512,
        embed_batch_size: int = 32,
        reranker_top_n: int = 5,
        chunk_size: int = 400,
        chunk_overlap: int = 80
    ):
        """
        Initialize the Advanced RAG System with specified configurations.

        Args:
            embedding_model_name: Name/path of the HuggingFace embedding model
            reranker_model_name: Name/path of the reranker model
            llm_model_name: Name of the Ollama LLM model to use
            device: Computing device ('cpu' or 'cuda')
            max_embedding_length: Maximum sequence length for embedding model
            embed_batch_size: Batch size for embedding generation
            reranker_top_n: Number of top results to return from reranker
            chunk_size: Size of document chunks for processing
            chunk_overlap: Overlap between consecutive document chunks
        """
        # Configuration parameters
        self.embedding_model_name = embedding_model_name
        self.reranker_model_name = reranker_model_name
        self.llm_model_name = llm_model_name
        self.device = device
        self.max_embedding_length = max_embedding_length
        self.embed_batch_size = embed_batch_size
        self.reranker_top_n = reranker_top_n
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Core components (will be initialized)
        self.embed_model = None
        self.reranker_model = None
        self.llm = None
        self.service_context = None
        self.node_parser = None
        self.index = None
        self.documents = []

        # Initialize system components
        self._initialize_components()
        logger.info("Advanced RAG System initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all core components of the RAG system"""
        try:
            self.embed_model = self._create_embedding_model()
            self.reranker_model = self._create_reranker_model()
            self.llm = self._create_llm()
            self.service_context = self._create_service_context()
            self.node_parser = self._create_node_parser()
        except Exception as e:
            logger.error(f"Failed to initialize system components: {str(e)}", exc_info=True)
            raise

    def _create_embedding_model(self) -> HuggingFaceEmbedding:
        """
        Create and configure the embedding model.

        Returns:
            Initialized HuggingFaceEmbedding instance
        """
        logger.info(f"Initializing embedding model: {self.embedding_model_name}")
        embed_args = {
            "model_name": self.embedding_model_name,
            "max_length": self.max_embedding_length,
            "embed_batch_size": self.embed_batch_size,
            "device": self.device,
        }
        return HuggingFaceEmbedding(** embed_args)

    def _create_reranker_model(self) -> BCERerank:
        """
        Create and configure the reranker model.

        Returns:
            Initialized BCERerank instance
        """
        logger.info(f"Initializing reranker model: {self.reranker_model_name}")
        reranker_args = {
            "model": self.reranker_model_name,
            "top_n": self.reranker_top_n,
            "device": self.device,
        }
        return BCERerank(**reranker_args)

    def _create_llm(self) -> Ollama:
        """
        Create and configure the Large Language Model (LLM).

        Returns:
            Initialized Ollama LLM instance
        """
        logger.info(f"Initializing LLM: {self.llm_model_name}")
        return Ollama(model=self.llm_model_name)

    def _create_service_context(self) -> ServiceContext:
        """
        Create the service context that bundles system components.

        Returns:
            Initialized ServiceContext instance
        """
        logger.info("Creating service context")
        return ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=self.embed_model
        )

    def _create_node_parser(self) -> SimpleNodeParser:
        """
        Create the document parser for splitting documents into chunks.

        Returns:
            Initialized SimpleNodeParser instance
        """
        logger.info(f"Creating node parser (chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})")
        return SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def load_documents(self, input_files: List[str] = None, input_dir: str = None) -> int:
        """
        Load documents from specified files or directory.

        Args:
            input_files: List of file paths to load
            input_dir: Directory containing files to load

        Returns:
            Number of documents loaded

        Raises:
            ValueError: If neither input_files nor input_dir is provided
        """
        if not input_files and not input_dir:
            raise ValueError("Either input_files or input_dir must be provided")

        try:
            reader_kwargs = {}
            if input_files:
                reader_kwargs["input_files"] = input_files
                logger.info(f"Loading documents from files: {input_files}")
            if input_dir:
                reader_kwargs["input_dir"] = input_dir
                logger.info(f"Loading documents from directory: {input_dir}")

            reader = SimpleDirectoryReader(** reader_kwargs)
            self.documents = reader.load_data()

            logger.info(f"Successfully loaded {len(self.documents)} documents")
            return len(self.documents)
        except Exception as e:
            logger.error(f"Error loading documents: {str(e)}", exc_info=True)
            raise

    def create_vector_index(self, max_documents: Optional[int] = None) -> VectorStoreIndex:
        """
        Create a vector store index from loaded documents.

        Args:
            max_documents: Maximum number of documents to use (None for all)

        Returns:
            Created VectorStoreIndex instance

        Raises:
            ValueError: If no documents have been loaded
        """
        if not self.documents:
            raise ValueError("No documents loaded. Call load_documents() first.")

        try:
            # Limit documents if specified
            docs_to_process = self.documents[:max_documents] if max_documents else self.documents
            logger.info(f"Processing {len(docs_to_process)} documents for index creation")

            # Split documents into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents(docs_to_process)
            logger.info(f"Parsed {len(nodes)} nodes from documents")

            # Create and store the index
            self.index = VectorStoreIndex(nodes, service_context=self.service_context)
            logger.info("Vector store index created successfully")

            return self.index
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}", exc_info=True)
            raise

    def generate_embeddings(
        self,
        query: str,
        passages: List[str]
    ) -> Tuple[List[float], List[List[float]]]:
        """
        Generate embeddings for a query and a list of passages.

        Args:
            query: Text query to generate embedding for
            passages: List of text passages to generate embeddings for

        Returns:
            Tuple containing query embedding and list of passage embeddings
        """
        try:
            logger.info(f"Generating embeddings for query: {query[:50]}...")

            # Generate embedding for the query
            query_embedding = self.embed_model.get_query_embedding(query)

            # Generate embeddings for all passages
            passage_embeddings = self.embed_model.get_text_embedding_batch(passages)

            logger.info(f"Successfully generated {len(passage_embeddings)} passage embeddings")
            return query_embedding, passage_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            raise

    def retrieve_and_rerank(
        self,
        query: str,
        similarity_top_k: int = 10
    ) -> List[NodeWithScore]:
        """
        Retrieve relevant document chunks using vector similarity and
        rerank them for improved relevance.

        Args:
            query: User query to retrieve relevant documents for
            similarity_top_k: Number of initial results from vector retrieval

        Returns:
            Reranked list of NodeWithScore objects

        Raises:
            ValueError: If vector index hasn't been created
        """
        if not self.index:
            raise ValueError("No vector index created. Call create_vector_index() first.")

        try:
            logger.info(f"Performing retrieval for query: {query}")

            # Initial retrieval using vector similarity
            vector_retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                service_context=self.service_context
            )
            initial_results = vector_retriever.retrieve(query)
            logger.info(f"Retrieved {len(initial_results)} initial results using vector similarity")

            # Rerank results for better relevance
            reranked_results = self.reranker_model.postprocess_nodes(
                initial_results,
                query_str=query
            )
            logger.info(f"Reranked results to top {len(reranked_results)}")

            return reranked_results
        except Exception as e:
            logger.error(f"Error during retrieval and reranking: {str(e)}", exc_info=True)
            raise

    def query_with_rag(self, query: str, similarity_top_k: int = 10) -> str:
        """
        Process a query using the complete RAG pipeline (retrieval + reranking + generation).

        Args:
            query: User query to process
            similarity_top_k: Number of initial results for retrieval

        Returns:
            Generated response to the query

        Raises:
            ValueError: If vector index hasn't been created
        """
        if not self.index:
            raise ValueError("No vector index created. Call create_vector_index() first.")

        try:
            logger.info(f"Processing RAG query: {query}")

            # Create query engine with reranker integration
            query_engine = self.index.as_query_engine(
                node_postprocessors=[self.reranker_model],
                similarity_top_k=similarity_top_k
            )

            # Get response from query engine
            response = query_engine.query(query)
            logger.info("Successfully generated response to query")

            return str(response)
        except Exception as e:
            logger.error(f"Error processing RAG query: {str(e)}", exc_info=True)
            raise

    def print_results(self, results: List[NodeWithScore], max_display_length: int = 200) -> None:
        """
        Print retrieval results in a human-readable format.

        Args:
            results: List of NodeWithScore objects to display
            max_display_length: Maximum length of text to display per result
        """
        print(f"\nRetrieved {len(results)} results:")
        print("=" * 100)

        for i, node in enumerate(results, 1):
            print(f"Result {i} (Score: {node.score:.4f}):")
            text = node.text
            if len(text) > max_display_length:
                text = text[:max_display_length] + "..."
            print(text)
            print("-" * 100)


def main():
    """Main function to demonstrate the Advanced RAG System functionality"""
    try:
        # Initialize the RAG system with default parameters
        rag_system = AdvancedRAGSystem(
            device="cpu",
            chunk_size=400,
            chunk_overlap=80,
            reranker_top_n=5
        )

        # Load documents from file
        rag_system.load_documents(input_files=["app.py"])

        # Create vector index from documents
        rag_system.create_vector_index(max_documents=36)

        # Example 1: Generate embeddings for a query and passages
        print("\n=== Example 1: Embedding Generation ===")
        query = "apples"
        passages = [
            "I like apples",
            "I like oranges",
            "Apples and oranges are fruits"
        ]

        query_embedding, passage_embeddings = rag_system.generate_embeddings(query, passages)
        print(f"Generated query embedding with length: {len(query_embedding)}")
        print(f"Generated embeddings for {len(passage_embeddings)} passages")

        # Example 2: Retrieve and rerank relevant documents
        print("\n=== Example 2: Retrieval and Reranking ===")
        query = "What is qwen 2?"
        reranked_results = rag_system.retrieve_and_rerank(query, similarity_top_k=10)

        # Print the top reranked results
        rag_system.print_results(reranked_results)

        # Example 3: Complete RAG query processing
        print("\n=== Example 3: Complete RAG Query ===")
        response = rag_system.query_with_rag(query)
        print(f"Query: {query}")
        print(f"Response: {response}")

    except Exception as e:
        logger.error(f"System error: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
