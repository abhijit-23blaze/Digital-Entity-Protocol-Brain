from brain.core import BrainRegion
from brain.schemas import BrainContext

class PFCPlanner(BrainRegion):
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
            # Naive parsing
            context.plan = [line.strip('- ') for line in raw_plan.split('\n') if line.strip().startswith('-') or line.strip().startswith('*')] 
            if not context.plan: context.plan = [raw_plan] # Fallback
            
            context.add_log(self.name, f"Plan Generated:\n{raw_plan}")
            context.current_stage = "Planned"
            
        return context
