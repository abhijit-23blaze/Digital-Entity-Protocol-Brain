from brain.core import BrainRegion
from brain.schemas import BrainContext

#TODO : instead of LLM calls need to actually train algo to do the routing based on EEG signals, but for now this should work

class PFCRouter:
    def __init__(self, llm_client, name="Prefrontal Cortex (Router)"):
        self.llm = llm_client
        self.name = name

    def decide_flow(self, query: str) -> tuple[str, str | None]:
        """Decides flow and optionally provides quick answer in one shot."""
        # self.llm.generate("Warmup", "Warmup") # Skip warmup for speed
        
        system_prompt = (
            "You are the Prefrontal Cortex, the brain's decision maker. "
            "Analyze the user query and select the best processing flow."
            "\nOptions:"
            "\n1. 'logical': For pure math, coding, facts, or scientific questions."
            "\n2. 'creative': For poetry, stories, art, or subjective topics."
            "\n3. 'sequential': For complex topics needing BOTH deep explanation and good writing."
            "\n4. 'parallel': For brainstorming or A/B testing diverse perspectives."
            "\n5. 'fast': For trivial greetings, simple factual questions, or queries not requiring deep thought. IF and ONLY IF you choose 'fast', provide the answer immediately in 'content'."
            "\n\nReturn a valid JSON object with keys: 'flow' (str) and 'content' (str or null)."
            "\nExample Fast: {\"flow\": \"fast\", \"content\": \"Hello! How can I help?\"}"
            "\nExample Logical: {\"flow\": \"logical\", \"content\": null}"
        )
        
        try:
            response = self.llm.generate(system_prompt, query, temperature=0.2).strip()
            # Clean up potential markdown code blocks
            if response.startswith("```"):
                response = response.strip("`").replace("json", "").strip()
            
            import json
            data = json.loads(response)
            flow = data.get("flow", "sequential").lower()
            content = data.get("content")
            
        except Exception as e:
            print(f"[Router Error] JSON Parse Failed: {e}. Defaulting to Sequential.")
            return "sequential", None
        
        # Validation
        valid_flows = ["logical", "creative", "sequential", "parallel", "fast"]
        if flow not in valid_flows: flow = "sequential"
        
        print(f"[Prefrontal Cortex] Decision: Routing to '{flow}' flow.")
        return flow, content


