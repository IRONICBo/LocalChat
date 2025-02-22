import gradio as gr
import pandas as pd
import chromadb

# from chromadb.utils import embedding_utils

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma")
# client = chromadb.Client(path="./chroma")
# Update with get or create
collection = client.get_or_create_collection("example_collection")


# Pagination for query results
def fetch_data(page_number, page_size):
    """
    Fetch data from ChromaDB with pagination.

    :param page_number: Current page number
    :param page_size: Number of items per page
    :return: DataFrame containing paginated records
    """
    all_data = collection.get()
    ids, embeddings, metadatas = (
        all_data["ids"],
        all_data["embeddings"],
        all_data["metadatas"],
    )

    # Create a DataFrame for easier handling and pagination
    data = pd.DataFrame({"ID": ids, "Embedding": embeddings, "Metadata": metadatas})

    # Pagination logic
    start = (page_number - 1) * page_size
    end = start + page_size
    return data.iloc[start:end].reset_index(drop=True)


# Upsert data into ChromaDB
def upsert_data(record_id, embedding, metadata):
    """
    Upsert (add or update) data in ChromaDB.

    :param record_id: Unique identifier of the record
    :param embedding: Embedding vector
    :param metadata: Additional metadata for the embedding
    :return: Confirmation message
    """
    embedding_vector = [float(i) for i in embedding.split(",")]
    metadata_dict = {"info": metadata}

    collection.upsert(
        ids=[record_id], embeddings=[embedding_vector], metadatas=[metadata_dict]
    )
    return f"Data with ID '{record_id}' has been upserted."


# Delete data from ChromaDB
def delete_data(record_id):
    """
    Delete data from ChromaDB.

    :param record_id: Unique identifier of the record to delete
    :return: Confirmation message
    """
    collection.delete(ids=[record_id])
    return f"Data with ID '{record_id}' has been deleted."


# Search in ChromaDB
def search_data(query_embedding, page_number, page_size):
    """
    Search data in ChromaDB based on embedding similarity.

    :param query_embedding: Embedding vector for similarity search
    :param page_number: Current page number for pagination
    :param page_size: Number of items per page
    :return: DataFrame of search results
    """
    query_vector = [float(i) for i in query_embedding.split(",")]

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=page_size * page_number,  # Fetch up to the nth result to paginate
    )

    # Extract results and paginate
    ids, scores, metadatas = (
        results["ids"][0],
        results["distances"][0],
        results["metadatas"][0],
    )
    data = pd.DataFrame({"ID": ids, "Score": scores, "Metadata": metadatas})

    # Pagination
    start = (page_number - 1) * page_size
    end = start + page_size
    return data.iloc[start:end].reset_index(drop=True)


# Gradio UI
def chromadb_manager_tab():
    gr.Markdown("## ChromaDB Manager")

    # Upsert Section
    record_id_input = gr.Textbox(label="Record ID")
    embedding_input = gr.Textbox(label="Embedding (comma-separated values)")
    metadata_input = gr.Textbox(label="Metadata")
    upsert_button = gr.Button("Upsert Data")
    upsert_result = gr.Textbox(label="Upsert Result")

    # Deletion Section
    delete_id_input = gr.Textbox(label="Record ID to Delete")
    delete_button = gr.Button("Delete Data")
    delete_result = gr.Textbox(label="Delete Result")

    # Query Section
    page_number_input = gr.Number(label="Page Number", value=1, precision=0)
    page_size_input = gr.Number(label="Page Size", value=10, precision=0)
    query_result_table = gr.DataFrame(
        headers=["ID", "Embedding", "Metadata"],
        datatype=["str", "str", "str"],
        label="Query Results",
        height=500,
    )
    fetch_data_button = gr.Button("Fetch Data")

    # Search Section
    search_embedding_input = gr.Textbox(
        label="Search Embedding (comma-separated values)"
    )
    search_page_number_input = gr.Number(
        label="Search Page Number", value=1, precision=0
    )
    search_page_size_input = gr.Number(label="Search Page Size", value=10, precision=0)
    search_result_table = gr.DataFrame(
        headers=["ID", "Score", "Metadata"],
        datatype=["str", "number", "str"],
        label="Search Results",
        height=500,
    )
    search_button = gr.Button("Search Data")

    # Button Click Events
    upsert_button.click(
        fn=upsert_data,
        inputs=[record_id_input, embedding_input, metadata_input],
        outputs=upsert_result,
    )
    delete_button.click(fn=delete_data, inputs=delete_id_input, outputs=delete_result)
    fetch_data_button.click(
        fn=fetch_data,
        inputs=[page_number_input, page_size_input],
        outputs=query_result_table,
    )
    search_button.click(
        fn=search_data,
        inputs=[
            search_embedding_input,
            search_page_number_input,
            search_page_size_input,
        ],
        outputs=search_result_table,
    )


# Main Gradio app
if __name__ == "__main__":
    with gr.Blocks() as main_block:
        gr.Markdown("<h1><center>ChromaDB Management System</center></h1>")

        with gr.Tabs():
            with gr.Tab(label="ChromaDB Manager"):
                chromadb_manager_tab()

    main_block.queue()
    main_block.launch()
