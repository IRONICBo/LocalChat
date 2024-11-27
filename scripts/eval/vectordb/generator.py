# TODO
import random
import re
import os
import fitz
import uuid
from typing import Dict, List, Tuple
from tqdm import tqdm
from llama_index.core.schema import BaseNode
from llama_index.core import Document, Settings
from llama_index.llms.base import BaseLLM
from llama_index.schema import MetadataType, TextNode
from llama_index.storage.docstore import DocStore
from llama_index.evaluation import EmbeddingFineTuneDataset
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms import Ollama

DEFAULT_PROMPT_TEMPLATE = """\
Below is the contextual information provided:

---------------------
{context_data}
---------------------

Using the given context only, without external knowledge, \
create a set of {questions_per_chunk} questions for a quiz or examination. \
The questions should cover a variety of aspects from the content. \
Do not provide answers, only questions.\"
"""

# Helper function to generate questions based on context
def create_question_context_pairs(
    text_nodes: List[TextNode],
    language_model: BaseLLM,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    questions_per_chunk: int = 2,
) -> EmbeddingFineTuneDataset:
    """Generate question-context pairs for fine-tuning."""
    # Extract text content from nodes
    content_map = {
        node.node_id: node.get_text(metadata_type=MetadataType.NONE)
        for node in text_nodes
    }

    question_map = {}
    context_relevance = {}
    for node_id, content in tqdm(content_map.items()):
        # Format the prompt for the LLM
        query_prompt = prompt_template.format(
            context_data=content, questions_per_chunk=questions_per_chunk
        )
        response = language_model.generate(query_prompt)

        # Parse the questions from the response
        parsed_questions = [
            re.sub(r"^\d+[\).\s]", "", line).strip()
            for line in response.strip().split("\n")
        ]
        parsed_questions = [q for q in parsed_questions if q]

        for question in parsed_questions:
            question_id = str(uuid.uuid4())
            question_map[question_id] = question
            context_relevance[question_id] = [node_id]

    # Construct the fine-tuning dataset
    return EmbeddingFineTuneDataset(
        queries=question_map, corpus=content_map, relevant_docs=context_relevance
    )

class QuestionGenerator:
    def __init__(
        self,
        llm_name="qwen2:0.5b",
    ) -> None:
        embed_args = {'model_name': 'maidalun1020/bce-embedding-base_v1', 'max_length': 512, 'embed_batch_size': 32, 'device': 'cpu'}
        self.embedding_model = HuggingFaceEmbedding(**embed_args)
        self.language_model = Ollama(model="qwen2:0.5b")
        self._processed_files = {}

    def _clean_text(self, text: str) -> str:
        """
        Cleans and normalizes text by applying regex patterns.
        """
        pattern = r'[a-zA-Z0-9 \u00C0-\u01B0\u1EA0-\u1EF9`~!@#$%^&*()_\-+=\[\]{}|\\;:\'",.<>/?]+'
        matches = re.findall(pattern, text)
        cleaned_text = ' '.join(matches)
        return re.sub(r'\s+', ' ', cleaned_text.strip())

    def _generate_questions_from_nodes(
        self,
        nodes,
        llm,
        prompt_template=DEFAULT_PROMPT_TEMPLATE,
        questions_per_chunk=2,
    ) -> EmbeddingFineTuneDataset:
        """
        Generates questions for fine-tuning using context from nodes.
        """
        content_map = {
            node.node_id: node.get_text(metadata_type=MetadataType.NONE)
            for node in nodes
        }
        question_map = {}
        relevance_map = {}

        for node_id, content in tqdm(content_map.items(), desc="Generating Questions"):
            prompt = prompt_template.format(
                context_data=content, questions_per_chunk=questions_per_chunk
            )
            response = llm.generate(prompt)

            questions = [
                re.sub(r"^\d+[\).\s]", "", q.strip())
                for q in response.strip().split("\n")
            ]
            questions = [q for q in questions if q]

            for question in questions:
                question_id = str(uuid.uuid4())
                question_map[question_id] = question
                relevance_map[question_id] = [node_id]

        return EmbeddingFineTuneDataset(
            queries=question_map, corpus=content_map, relevant_docs=relevance_map
        )

    def process_and_store(
        self,
        input_files,
        embed=True,
        embed_model=None,
    ) -> List[BaseNode]:
        """
        Reads, processes, and stores nodes from input files.
        """
        nodes_to_return = []
        splitter = SentenceSplitter.from_defaults(
            chunk_size=1024,
            chunk_overlap=1024,
            paragraph_separator=1024,
            secondary_chunking_regex=1024
        )

        for input_file in tqdm(input_files, desc="Processing and Ingesting Files"):
            file_name = os.path.basename(input_file)

            if file_name in self._processed_files:
                nodes_to_return.extend(self._processed_files[file_name])
                continue

            # Extract text from the document
            document = fitz.open(input_file)
            all_text = ""
            for page in document:
                page_text = page.get_text("text")
                cleaned_text = self._clean_text(page_text)
                all_text += " " + cleaned_text

            document_obj = Document(
                text=all_text.strip(),
                metadata={"file_name": file_name},
            )

            # Split into nodes and optionally embed
            nodes = splitter([document_obj], show_progress=True)
            if embed:
                nodes = Settings.embed_model(nodes, show_progress=True)

            self._processed_files[file_name] = nodes
            nodes_to_return.extend(nodes)

        return nodes_to_return

    def generate_questions(
        self,
        input_files: List[str],
        output_directory: str = "output_dataset",
        max_samples: int = 100,
        questions_per_chunk: int = 2,
    ) -> None:
        """
        Generates a dataset of questions and stores it along with nodes.
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        if os.path.exists(os.path.join(output_directory, "docstore.json")):
            print("Dataset already exists. Skipping processing.")
            return

        # Process files and create nodes
        nodes = self.process_and_store(input_files, embed=True)
        random.shuffle(nodes)

        # Generate question-context dataset
        dataset = self._generate_questions_from_nodes(
            nodes[:max_samples], llm=self.language_model, questions_per_chunk=questions_per_chunk
        )

        # Save dataset and nodes
        dataset.save_to_file(os.path.join(output_directory, "dataset.json"))
        doc_store = DocStore()
        doc_store.add_documents(nodes)
        doc_store.save(os.path.join(output_directory, "docstore.json"))