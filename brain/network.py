from brain.Prefrontal_Cortex import PrefrontalCortex
from brain.Hippocampus.memory import Hippocampus
from brain.Left_Hemisphere.logic import LeftHemisphere
from brain.Right_Hemisphere.creative import RightHemisphere
from brain.core import LLMClient
from brain.schemas import BrainContext
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BrainNetwork:
    def __init__(self):
        # Each brain region gets a model optimized for its role
        pfc_llm   = LLMClient(model_name='gemini-3-flash-preview')             # Fast planning & synthesis
        left_llm  = LLMClient(model_name='gemini-3-pro-preview', thinking='high')    # Deep reasoning
        right_llm = LLMClient(model_name='gemini-3-pro-preview', thinking='minimal') # Creative, no deep reasoning
        hippo_llm = LLMClient(model_name='gemini-3-flash-preview')             # Memory retrieval

        self.pfc   = PrefrontalCortex("Prefrontal Cortex", pfc_llm)
        self.hippo = Hippocampus("Hippocampus", hippo_llm)
        self.left  = LeftHemisphere("Left Hemisphere", left_llm)
        self.right = RightHemisphere("Right Hemisphere", right_llm)

    async def run_dynamic(self, query: str) -> BrainContext:
        """Asks the PFC to decide the flow, then executes it."""
        flow, content = self.pfc.decide_flow(query)
        
        if flow == "fast":
            # Pass the pre-generated content to run_fast
            return self.run_fast(query, content)
        elif flow == "logical":
            return self.run_logical(query)
        elif flow == "creative":
            return self.run_creative(query)
        elif flow == "parallel":
            # Parallel is async
            return await self.run_parallel(query)
        else:
            # Default to sequential
            return self.run_sequential(query)

    def run_fast(self, query: str, content: str = None) -> BrainContext:
        """Flow E: Fast / Trivial"""
        ctx = BrainContext(original_query=query, current_stage="Fast Flow")
        # Reuse quick_reply but passing the content if we have it
        ctx = self.pfc.quick_reply(ctx, content) 
        return ctx

    def run_sequential(self, query: str) -> BrainContext:
        """Flow A: The Waterfall"""
        ctx = BrainContext(original_query=query, current_stage="Sequential Flow")
        ctx = self.pfc.process(ctx)
        ctx = self.hippo.process(ctx)
        ctx = self.left.process(ctx)
        self._index_search_results(ctx)
        ctx = self.right.process(ctx)
        return ctx

    def run_logical(self, query: str) -> BrainContext:
        """Flow C: Pure Logic"""
        ctx = BrainContext(original_query=query, current_stage="Logical Flow")
        ctx = self.pfc.process(ctx)
        ctx = self.hippo.process(ctx)
        ctx = self.left.process(ctx)
        self._index_search_results(ctx)
        
        # Final output is the Logical facts joined
        ctx.final_output = "\n".join(ctx.logical_facts) if ctx.logical_facts else "No logical output generated."
        return ctx

    def run_creative(self, query: str) -> BrainContext:
        """Flow D: Pure Creativity"""
        ctx = BrainContext(original_query=query, current_stage="Creative Flow")
        ctx = self.pfc.process(ctx)
        ctx = self.hippo.process(ctx)
        ctx = self.right.process(ctx)
        return ctx

    async def run_parallel(self, query: str) -> BrainContext:
        """Flow B: The Parallel Council"""
        ctx = BrainContext(original_query=query, current_stage="Parallel Flow")
        
        ctx = self.pfc.process(ctx)
        ctx = self.hippo.process(ctx)
        
        with ThreadPoolExecutor() as executor:
            loop = asyncio.get_event_loop()
            t1 = loop.run_in_executor(executor, self.left.process, ctx.model_copy(deep=True))
            t2 = loop.run_in_executor(executor, self.right.process, ctx.model_copy(deep=True))
            
            ctx_left, ctx_right = await asyncio.gather(t1, t2)
            
        ctx.logical_facts = ctx_left.logical_facts
        ctx.search_results = ctx_left.search_results
        ctx.creative_draft = ctx_right.creative_draft
        ctx.logs.extend(ctx_left.logs)
        ctx.logs.extend(ctx_right.logs)
        
        # Index search results discovered during parallel processing
        self._index_search_results(ctx)
        
        # PFC Synthesis
        ctx = self.pfc.process(ctx)
        return ctx

    def _index_search_results(self, ctx: BrainContext):
        """Index Tavily search results into HippoRAG for long-term memory."""
        if ctx.search_results:
            try:
                self.hippo.index_search_results(ctx.search_results)
                ctx.add_log("BrainNetwork", f"Indexed {len(ctx.search_results)} search results into long-term memory.")
            except Exception as e:
                ctx.add_log("BrainNetwork", f"Failed to index search results: {e}")
