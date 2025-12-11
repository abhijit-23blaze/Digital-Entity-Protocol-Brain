from brain.Prefrontal_Cortex.router import PFCRouter
from brain.Prefrontal_Cortex.planner import PFCPlanner
from brain.core import BrainRegion, LLMClient
from brain.schemas import BrainContext

class PrefrontalCortex(BrainRegion):
    def __init__(self, name: str, llm: LLMClient):
        super().__init__(name, llm)
        self.router = PFCRouter(llm, name)
        self.planner = PFCPlanner("Prefrontal Cortex (Planner)", llm)

    def decide_flow(self, query: str) -> str:
        return self.router.decide_flow(query)

    def quick_reply(self, context: BrainContext) -> BrainContext:
        return self.router.quick_reply(context)

    def process(self, context: BrainContext) -> BrainContext:
        return self.planner.process(context)
