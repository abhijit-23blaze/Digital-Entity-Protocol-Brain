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

class RightHemisphere(BrainRegion):
    def process(self, context: BrainContext) -> BrainContext:
        context.add_log(self.name, "Synthesizing creative output...")
        
        system_prompt = (
            "You are the Right Hemisphere, the Artist. "
            "You value emotion, style, and abstraction. "
            "Take the dry facts from the Left Hemisphere (if available) or the raw query and rewrite them. "
            "Make it human, poetic, or engaging."
        )
        
        if context.logical_facts:
            # Sequential mode input
            data = f"Query: {context.original_query}\nDry Facts: {context.logical_facts}"
        else:
             # Parallel mode input (raw query + plan)
             # OR Creative Mode input (Logical facts might be None)
             data = f"Query: {context.original_query}\nPlan: {context.plan}\nContext: {context.memories}"

        # High temperature for creativity
        context.final_output = self.llm.generate(system_prompt, data, temperature=0.9)
        # Store as draft if we are in parallel mode (PFC will synthesize later) - detecting by caller but simpler to just store in final_output for sequential
        context.creative_draft = context.final_output
        
        # In sequential, this is the final, but we log it as the Right Brain's work
        context.add_log(self.name, f"Creative Output:\n{context.final_output }")
        
        context.current_stage = "Creating (Art)"
        return context
