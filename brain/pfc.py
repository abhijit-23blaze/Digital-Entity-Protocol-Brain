from brain.core import BrainRegion
from brain.schemas import BrainContext
import json

class PrefrontalCortex(BrainRegion):
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
        
        # We print this decision manually since it happens before the main context loop
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

    def process(self, context: BrainContext) -> BrainContext:
        context.add_log(self.name, "Analyzing query and creating plan...")
        
        if context.logical_facts and context.creative_draft:
             # Synthesis Mode (Parallel Flow End)
             system_prompt = (
                "You are the Prefrontal Cortex, the executive center. "
                "You have received two inputs: a pure logical analysis (Left Brain) and a creative draft (Right Brain). "
                "Synthesize them into one perfect, balanced response. "
                "Maintain the accuracy of the Left but the engagement of the Right."
            )
             user_content = f"Query: {context.original_query}\n\nLogical Data: {context.logical_facts}\n\nCreative Draft: {context.creative_draft}"
             context.final_output = self.llm.generate(system_prompt, user_content)
             context.current_stage = "Synthesis Complete"
        
        else:
            # Planning Mode
            system_prompt = (
                "You are the Prefrontal Cortex (PFC), the Architect of this brain. "
                "Your job is NOT to answer the user, but to PLAN. "
                "Break down the user's query into a list of 3-5 clear, actionable steps for the other brain regions. "
                "Return valid JSON formatted list of strings."
            )
            raw_plan = self.llm.generate(system_prompt, context.original_query)
            context.plan = [line.strip('- ') for line in raw_plan.split('\n') if line.strip().startswith('-') or line.strip().startswith('*')] 
            if not context.plan: context.plan = [raw_plan] # Fallback
            
            context.add_log(self.name, f"Plan Generated:\n{raw_plan}")
            context.current_stage = "Planned"
            
        return context
