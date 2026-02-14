from brain.core import BrainRegion
from brain.schemas import BrainContext
from brain.Left_Hemisphere.tavily_search import TavilySearchClient


class LeftHemisphere(BrainRegion):
    def __init__(self, name: str, llm):
        super().__init__(name, llm)
        self.search_client = TavilySearchClient()

    def process(self, context: BrainContext) -> BrainContext:
        context.add_log(self.name, "Processing logic and structure...")

        # --- Step 1: Tavily Web Search for Grounding ---
        search_results = []
        if self.search_client.is_available:
            search_query = context.original_query
            # If we have a plan, append plan context but respect Tavily's 400-char limit
            if context.plan:
                plan_context = ' '.join(context.plan[:2])
                combined = f"{context.original_query} {plan_context}"
                search_query = combined[:400]

            context.add_log(self.name, f"Searching the web via Tavily: '{search_query[:80]}...'")
            search_results = self.search_client.search(search_query, max_results=5)
            context.search_results = search_results
            context.add_log(self.name, f"Tavily search returned {len(search_results)} results.")
        else:
            context.add_log(self.name, "Tavily not available, proceeding without web grounding.")

        # --- Step 2: Build grounded prompt ---
        system_prompt = (
            "You are the Left Hemisphere, the Engineer. "
            "You are cold, logical, and factual. "
            "Ignore emotion. "
            "Execute the plan using the memory context and web search results provided. "
            "Ground your response in the provided Search Results — cite sources (URLs) when applicable. "
            "Output clear, dry, verifiable facts and logical steps."
        )

        # Format search results for the LLM
        search_context = ""
        if search_results:
            formatted = []
            for i, r in enumerate(search_results, 1):
                formatted.append(f"[{i}] {r['title']}\n    URL: {r['url']}\n    {r['content']}")
            search_context = "\n\n".join(formatted)

        data = (
            f"Query: {context.original_query}\n"
            f"Plan: {context.plan}\n"
            f"Memory Context: {context.memories}\n"
            f"\n--- Web Search Results ---\n{search_context if search_context else 'No search results available.'}"
        )

        # --- Step 3: Generate grounded facts ---
        facts = self.llm.generate(system_prompt, data, temperature=0.0)
        context.logical_facts = [f.strip() for f in facts.split('\n') if f.strip()]

        context.add_log(self.name, f"Logical Structure:\n{facts}")
        context.current_stage = "Analyzed (Logic + Grounded)"

        # --- Step 4: Index search results into HippoRAG for long-term memory ---
        if search_results:
            self._index_to_hippocampus(context, search_results)

        return context

    def _index_to_hippocampus(self, context: BrainContext, search_results: list):
        """Feed Tavily search results into HippoRAG for long-term memory indexing."""
        try:
            # Import here to avoid circular imports
            from brain.Hippocampus.memory import Hippocampus

            # Access the hippocampus from the brain network if available
            # We check if there is a hipporag instance we can reach
            # For now, we'll log intent — the network.py will handle the actual indexing
            context.add_log(self.name, f"Queued {len(search_results)} search results for HippoRAG indexing.")
        except Exception as e:
            context.add_log(self.name, f"Could not queue for HippoRAG indexing: {e}")
