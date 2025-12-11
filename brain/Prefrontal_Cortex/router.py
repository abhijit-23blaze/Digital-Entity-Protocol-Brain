from brain.core import BrainRegion
from brain.schemas import BrainContext

class PFCRouter:
    def __init__(self, llm_client, name="Prefrontal Cortex (Router)"):
        self.llm = llm_client
        self.name = name

    def decide_flow(self, query: str) -> str:
        """Decides the best processing flow based on the query."""
        self.llm.generate("Warmup", "Warmup") # Wake up call
        
        system_prompt = (
            "You are the Prefrontal Cortex, the brain's decision maker. "
            "Analyze the user query and select the best processing flow."
            "\nOptions:"
            "\n1. 'logical': For pure math, coding, facts, or scientific questions. (Path: PFC->Hippocampus->Left)"
            "\n2. 'creative': For poetry, stories, art, or subjective topics. (Path: PFC->Hippocampus->Right)"
            "\n3. 'sequential': For complex topics needing BOTH deep explanation and good writing. (Path: PFC->Hippocampus->Left->Right)"
            "\n4. 'parallel': For brainstorming or A/B testing diverse perspectives. (Path: Parallel Left/Right)"
            "\n5. 'fast': For trivial greetings, simple factual questions, or queries not requiring deep thought. (Path: PFC Only)"
            "\n\nReturn ONLY the flow name (logical, creative, sequential, parallel, or fast) in lowercase."
        )
        
        flow = self.llm.generate(system_prompt, query, temperature=0.2).strip().lower()
        
        # Validation/Fallback
        valid_flows = ["logical", "creative", "sequential", "parallel", "fast"]
        cleaned_flow = next((f for f in valid_flows if f in flow), "sequential")
        
        print(f"[Prefrontal Cortex] Decision: Routing to '{cleaned_flow}' flow.")
        return cleaned_flow

    def quick_reply(self, context: BrainContext) -> BrainContext:
        """Standard direct LLM response, bypassing other regions."""
        context.add_log(self.name, "Query deemed trivial. Responding directly...")
        
        system_prompt = (
            "You are the Prefrontal Cortex. "
            "The user asked a simple question that does not require the full brain network. "
            "Answer directly, concisely, and helpfully."
        )
        context.final_output = self.llm.generate(system_prompt, context.original_query)
        context.current_stage = "Quick Response"
        return context
