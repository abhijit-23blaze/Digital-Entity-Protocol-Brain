from brain.Prefrontal_Cortex.routing_engine import PFCRouter
from brain.Prefrontal_Cortex.planner import PFCPlanner
from brain.core import BrainRegion, LLMClient
from brain.schemas import BrainContext

class PrefrontalCortex(BrainRegion):
    def __init__(self, name: str, llm: LLMClient):
        super().__init__(name, llm)
        
        # Creating a specialized faster client for the router
        router_llm = LLMClient(model_name='gemini-3-flash-preview')
        
        self.router = PFCRouter(router_llm, name)
        self.planner = PFCPlanner("Prefrontal Cortex (Planner)", llm)

    def decide_flow(self, query: str) -> tuple[str, str]:
        return self.router.decide_flow(query)

    def quick_reply(self, context: BrainContext, content: str = None) -> BrainContext:
        if content:
             # Optimization: Use the content we already got from the router
             context.add_log(self.router.name, "Fast response generated during routing.")
             context.final_output = content
             context.current_stage = "Quick Response (Cached)"
             return context
        # Fallback if content wasn't cached (e.g. manual CLI usage)
        context.add_log(self.router.name, "Query deemed trivial. Responding directly (Fallback)...")
        system_prompt = (
            "You are the Prefrontal Cortex. "
            "The user asked a simple question that does not require the full brain network. "
            "Answer directly, concisely, and helpfully."
        )
        # We access the router's efficient LLM directly
        context.final_output = self.router.llm.generate(system_prompt, context.original_query)
        context.current_stage = "Quick Response (Fallback)"
        return context

    def process(self, context: BrainContext) -> BrainContext:
        return self.planner.process(context)
