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
        self.llm = LLMClient()
        self.pfc = PrefrontalCortex("Prefrontal Cortex", self.llm)
        self.hippo = Hippocampus("Hippocampus", self.llm)
        self.left = LeftHemisphere("Left Hemisphere", self.llm)
        self.right = RightHemisphere("Right Hemisphere", self.llm)

    async def run_dynamic(self, query: str) -> BrainContext:
        """Asks the PFC to decide the flow, then executes it."""
        flow = self.pfc.decide_flow(query)
        
        if flow == "fast":
            return self.run_fast(query)
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

    def run_fast(self, query: str) -> BrainContext:
        """Flow E: Fast / Trivial"""
        ctx = BrainContext(original_query=query, current_stage="Fast Flow")
        # PFC handles everything directly
        ctx = self.pfc.quick_reply(ctx) 
        return ctx

    def run_sequential(self, query: str) -> BrainContext:
        """Flow A: The Waterfall"""
        ctx = BrainContext(original_query=query, current_stage="Sequential Flow")
        ctx = self.pfc.process(ctx)
        ctx = self.hippo.process(ctx)
        ctx = self.left.process(ctx)
        ctx = self.right.process(ctx)
        return ctx

    def run_logical(self, query: str) -> BrainContext:
        """Flow C: Pure Logic"""
        ctx = BrainContext(original_query=query, current_stage="Logical Flow")
        ctx = self.pfc.process(ctx)
        ctx = self.hippo.process(ctx)
        ctx = self.left.process(ctx)
        
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
        ctx.creative_draft = ctx_right.creative_draft
        ctx.logs.extend(ctx_left.logs)
        ctx.logs.extend(ctx_right.logs)
        
        # PFC Synthesis
        ctx = self.pfc.process(ctx)
        return ctx
