import sys
import os
from brain.core import BrainRegion
from brain.schemas import BrainContext

# Add HippoRAG src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
hipporag_src_path = os.path.join(current_dir, "HippoRAG", "src")
if hipporag_src_path not in sys.path:
    sys.path.append(hipporag_src_path)

try:
    from hipporag import HippoRAG
except ImportError as e:
    print(f"Warning: Failed to import HippoRAG: {e}")
    HippoRAG = None

class Hippocampus(BrainRegion):
    def __init__(self, name: str, llm):
        super().__init__(name, llm)
        self.hipporag = None
        if HippoRAG:
            try:
                # Initialize HippoRAG with Gemini
                # We use a memory_storage subfolder for keeping indices
                save_dir = os.path.join(current_dir, "memory_storage")
                # Using Gemini 1.5 Flash as it's fast and effective
                self.hipporag = HippoRAG(
                    save_dir=save_dir,
                    llm_model_name="gemini-1.5-flash", 
                    embedding_model_name="gemini-embedding"
                )
            except Exception as e:
                print(f"Error initializing HippoRAG: {e}")

    def process(self, context: BrainContext) -> BrainContext:
        context.add_log(self.name, "Retrieving relevant context/memories...")
        
        if not self.hipporag:
            context.add_log(self.name, "HippoRAG not initialized, skipping retrieval.")
             # Fallback to simple generation if HippoRAG fails? 
             # Or just return empty. Original code had LLM generation. 
             # Let's keep a fallback if HippoRAG is down? 
             # No, user wants to use HippoRAG.
            return context

        query = "\n".join(context.plan) if context.plan else context.original_query
        
        try:
            # self.hipporag.retrieve returns a list of QuerySolution
            # We treat the query as a list of 1 string
            results = self.hipporag.retrieve(queries=[query], num_to_retrieve=2)
            
            memories = []
            if results and len(results) > 0:
                memories = results[0].docs
            
            context.memories = memories
            context.add_log(self.name, f"Context Retrieved:\n{memories}")
            
        except Exception as e:
            context.add_log(self.name, f"Error during retrieval: {e}")
            # Fallback to LLM simulation if RAG fails (optional, but good for robustness)
            # context.add_log(self.name, "Fallback to LLM simulation.")
            # ... (omitted for now to focus on getting RAG working)

        context.current_stage = "Contextualized"
        return context
        
    def add_memory(self, content: str):
        """Allows adding new memories (documents) to the RAG store."""
        if self.hipporag:
            # Index expects a list of docs
            self.hipporag.index(docs=[content])
            return True
        return False

    def index_search_results(self, search_results: list):
        """
        Index Tavily search results into HippoRAG for long-term memory.
        
        Each search result's content becomes a document in the knowledge graph,
        prefixed with its title and source URL for traceability.
        """
        if not self.hipporag or not search_results:
            return False

        docs = []
        for result in search_results:
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            content = result.get("content", "")
            if content.strip():
                doc = f"[Source: {title} | {url}]\n{content}"
                docs.append(doc)

        if docs:
            try:
                self.hipporag.index(docs=docs)
                print(f"[Hippocampus] Indexed {len(docs)} search results into HippoRAG.")
                return True
            except Exception as e:
                print(f"[Hippocampus] Failed to index search results: {e}")

        return False

