from brain.core import BrainRegion
from brain.schemas import BrainContext

class LeftHemisphere(BrainRegion):
    def process(self, context: BrainContext) -> BrainContext:
        context.add_log(self.name, "Processing logic and structure...")
        system_prompt = (
            "You are the Left Hemisphere, the Engineer. "
            "You are cold, logical, and factual. "
            "Ignore emotion. "
            "Execute the plan using the memory context provided. "
            "Output clear, dry, verifiable facts and logical steps."
        )
        data = f"Query: {context.original_query}\nPlan: {context.plan}\nContext: {context.memories}"
        facts = self.llm.generate(system_prompt, data, temperature=0.0) # High precision for logic
        context.logical_facts = [f.strip() for f in facts.split('\n') if f.strip()]
        
        context.add_log(self.name, f"Logical Structure:\n{facts}")
        context.current_stage = "Analyzed (Logic)"
        return context
