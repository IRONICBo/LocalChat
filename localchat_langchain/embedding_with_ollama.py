import os
import logging
from typing import List, Optional, Tuple, Dict
from llama_index.embeddings import HuggingFaceEmbedding, OllamaEmbedding
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
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("rag_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedRAGSystem")

class AdvancedRAGSystem:
    def __init__(
        self,
        embedding_model_name: str = "nomic-embed-text:latest",
        llm_model_name: str = "qwen2:0.5b",
        use_ollama: bool = True,
        device: str = "cpu",
        chunk_size: int = 400,
        chunk_overlap: int = 80
    ):
        self.embedding_model_name = embedding_model_name
        self.llm_model_name = llm_model_name
        self.use_ollama = use_ollama
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embed_model = self._init_embedding_model()
        self.llm = self._init_llm()
        self.service_context = self._init_service_context()
        self.node_parser = self._init_node_parser()

        self.documents: List[Document] = []
        self.index: Optional[VectorStoreIndex] = None

        logger.info("Advanced RAG System initialized successfully")

    def _init_embedding_model(self) -> HuggingFaceEmbedding | OllamaEmbedding:
        try:
            if self.use_ollama:
                logger.info(f"Initializing Ollama embedding model: {self.embedding_model_name}")
                return OllamaEmbedding(model_name=self.embedding_model_name)
            else:
                logger.info(f"Initializing HuggingFace embedding model: {self.embedding_model_name}")
                embed_args = {
                    "model_name": self.embedding_model_name,
                    "max_length": 512,
                    "embed_batch_size": 32,
                    "device": self.device,
                }
                return HuggingFaceEmbedding(** embed_args)
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}", exc_info=True)
            raise

    def _init_llm(self) -> Ollama:
        try:
            logger.info(f"Initializing LLM: {self.llm_model_name}")
            return Ollama(model=self.llm_model_name)
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}", exc_info=True)
            raise

    def _init_service_context(self) -> ServiceContext:
        try:
            logger.info("Creating service context")
            return ServiceContext.from_defaults(
                llm=self.llm,
                embed_model=self.embed_model
            )
        except Exception as e:
            logger.error(f"Failed to create service context: {str(e)}", exc_info=True)
            raise

    def _init_node_parser(self) -> SimpleNodeParser:
        try:
            logger.info(f"Creating node parser with chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
            return SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
        except Exception as e:
            logger.error(f"Failed to create node parser: {str(e)}", exc_info=True)
            raise

    def load_documents(self, input_files: List[str] = None, input_dir: str = None) -> int:
        try:
            if not input_files and not input_dir:
                raise ValueError("Either input_files or input_dir must be provided")

            reader_kwargs = {}
            if input_files:
                reader_kwargs["input_files"] = input_files
                logger.info(f"Loading documents from files: {input_files}")
            if input_dir:
                reader_kwargs["input_dir"] = input_dir
                logger.info(f"Loading documents from directory: {input_dir}")

            reader = SimpleDirectoryReader(**reader_kwargs)
            self.documents = reader.load_data()

            logger.info(f"Successfully loaded {len(self.documents)} documents")
            return len(self.documents)
        except Exception as e:
            logger.error(f"Failed to load documents: {str(e)}", exc_info=True)
            raise

    def create_index(self, max_documents: Optional[int] = None) -> VectorStoreIndex:
        try:
            if not self.documents:
                raise ValueError("No documents loaded. Call load_documents first.")

            docs_to_process = self.documents[:max_documents] if max_documents else self.documents
            logger.info(f"Processing {len(docs_to_process)} documents for index creation")

            nodes = self.node_parser.get_nodes_from_documents(docs_to_process)
            logger.info(f"Parsed {len(nodes)} nodes from documents")

            self.index = VectorStoreIndex(nodes, service_context=self.service_context)
            logger.info("Index created successfully")

            return self.index
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}", exc_info=True)
            raise

    def generate_embeddings(
        self,
        query: str,
        passages: List[str]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        try:
            logger.info(f"Generating embeddings for query: {query[:50]}...")

            query_embedding = self.embed_model.get_query_embedding(query)

            passages_embeddings = self.embed_model.get_text_embedding_batch(passages)

            logger.info(f"Generated {len(passages_embeddings)} passage embeddings")
            return query_embedding, passages_embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise

    def retrieve(
        self,
        query: str,
        similarity_top_k: int = 10,
        use_reranker: bool = False,
        reranker_top_n: int = 5
    ) -> List[NodeWithScore]:
        try:
            if not self.index:
                raise ValueError("No index created. Call create_index first.")

            logger.info(f"Retrieving for query: {query} with similarity_top_k={similarity_top_k}")

            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=similarity_top_k,
                service_context=self.service_context
            )

            retrieval_results = retriever.retrieve(query)
            logger.info(f"Retrieved {len(retrieval_results)} results using vector similarity")

            if use_reranker:
                logger.warning("Reranker is not implemented in this version")
                # from BCEmbedding.tools.llama_index import BCERerank
                # reranker = BCERerank(
                #     model="maidalun1020/bce-reranker-base_v1",
                #     top_n=reranker_top_n,
                #     device=self.device
                # )
                # reranked_results = reranker.postprocess_nodes(retrieval_results, query_str=query)
                # logger.info(f"Reranked results to top {reranker_top_n}")
                # return reranked_results

            return retrieval_results
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
            raise

    def query(self, query: str, similarity_top_k: int = 10) -> str:
        try:
            if not self.index:
                raise ValueError("No index created. Call create_index first.")

            logger.info(f"Processing query: {query}")

            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                service_context=self.service_context
            )

            response = query_engine.query(query)

            logger.info("Query processed successfully")
            return str(response)
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}", exc_info=True)
            raise

    def print_retrieval_results(self, results: List[NodeWithScore], max_length: int = 200):
        print(f"\nRetrieved {len(results)} results:")
        print("=" * 80)

        for i, node in enumerate(results, 1):
            print(f"Result {i} (Score: {node.score:.4f}):")
            text = node.text
            if len(text) > max_length:
                text = text[:max_length] + "..."
            print(text)
            print("-" * 80)


def main():
    try:
        rag_system = AdvancedRAGSystem(
            embedding_model_name="nomic-embed-text:latest",
            llm_model_name="qwen2:0.5b",
            chunk_size=400,
            chunk_overlap=80
        )

        rag_system.load_documents(input_files=["app.py"])

        rag_system.create_index(max_documents=36)

        print("\n=== Example 1: Generating Embeddings ===")
        query = "apples"
        passages = [
            "I like apples",
            "I like oranges",
            "Apples and oranges are fruits"
        ]

        query_embedding, passage_embeddings = rag_system.generate_embeddings(query, passages)
        print(f"Generated query embedding of length: {len(query_embedding)}")
        print(f"Generated {len(passage_embeddings)} passage embeddings")

        print("\n=== Example 2: Performing Retrieval ===")
        query = "What is qwen 2?"
        retrieval_results = rag_system.retrieve(
            query,
            similarity_top_k=5,
            use_reranker=False
        )

        rag_system.print_retrieval_results(retrieval_results)

        print("\n=== Example 3: Getting Answer ===")
        answer = rag_system.query(query)
        print(f"Answer to query '{query}':")
        print(answer)

    except Exception as e:
        logger.error(f"An error occurred in main: {str(e)}", exc_info=True)
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
