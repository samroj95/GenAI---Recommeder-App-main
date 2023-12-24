import os
from haystack import Document, Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever.multimodal import MultiModalRetriever

class MultimodalSearch:
    EMBEDDING_DIM = 512
    QUERY_MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
    DOC_DIR = "Data"
    
    def __init__(self):
        self.document_store = self._initialize_document_store()
        self.retriever_text_to_image = self._initialize_retriever()
        self.pipeline = self._initialize_pipeline()
        
    def _initialize_document_store(self):
        """Initialize the document store with a specified embedding dimension."""
        return InMemoryDocumentStore(embedding_dim=self.EMBEDDING_DIM)
    
    def _load_images(self):
        """Load image documents from the specified directory."""
        return [
            Document(content=f"./{self.DOC_DIR}/{filename}", content_type="image")
            for filename in os.listdir(f"./{self.DOC_DIR}")
        ]
    
    def _initialize_retriever(self):
        """Initialize the multimodal retriever."""
        retriever = MultiModalRetriever(
            document_store=self.document_store,
            query_embedding_model=self.QUERY_MODEL_NAME,
            query_type="text",
            document_embedding_models={"image": self.QUERY_MODEL_NAME},
        )
        
        # Turn images into embeddings and store them in the DocumentStore
        images = self._load_images()
        self.document_store.write_documents(images)
        self.document_store.update_embeddings(retriever=retriever)
        
        return retriever
    
    def _initialize_pipeline(self):
        """Initialize the pipeline with the retriever node."""
        pipeline = Pipeline()
        pipeline.add_node(component=self.retriever_text_to_image, name="retriever_text_to_image", inputs=["Query"])
        return pipeline
    
    def search(self, query, top_k=3):
        """Search the pipeline and return the top k results."""
        results = self.pipeline.run(query=query, params={"retriever_text_to_image": {"top_k": top_k}})
        return sorted(results["documents"], key=lambda d: d.score, reverse=True)
