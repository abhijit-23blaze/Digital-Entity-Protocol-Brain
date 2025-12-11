from brain.core import BrainRegion
from brain.schemas import BrainContext

class Hippocampus(BrainRegion):
    def process(self, context: BrainContext) -> BrainContext:
        context.add_log(self.name, "Retrieving relevant context/memories...")
        system_prompt = (
            "You are the Hippocampus, the Librarian. "
            "You do not reason; you remember. "
            "Given the plan, list 2 key historical concepts, analogies, or context items that would help. "
            "Return a concise list."
        )
        plan_str = "\n".join(context.plan) if context.plan else context.original_query
        memories = self.llm.generate(system_prompt, f"Plan: {plan_str}")
        context.memories = [m.strip() for m in memories.split('\n') if m.strip()]
        
        context.add_log(self.name, f"Context Retrieved:\n{memories}")
        context.current_stage = "Contextualized"
        return context
