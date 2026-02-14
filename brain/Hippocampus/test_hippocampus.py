import sys
import os

# Adust path to include project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from brain.Hippocampus.memory import Hippocampus
from brain.schemas import BrainContext

def test_hippocampus():
    print("Initializing Hippocampus...")
    # Mock LLM for init (Hippocampus takes it but doesn't use it for RAG anymore with our change)
    class MockLLM:
        pass
    
    hippo = Hippocampus("Hippocampus", MockLLM())
    
    if not hippo.hipporag:
        print("Failed to initialize HippoRAG inside Hippocampus.")
        return

    print("Adding memory...")
    # Add a memory
    memory_content = "The user, Abhi, is working on a digital brain project using Gemini and HippoRAG."
    hippo.add_memory(memory_content)
    
    print("Memory added. Now testing retrieval...")
    
    # Create a context with a plan/query
    ctx = BrainContext(original_query="What is Abhi working on?", plan=["Retrieve info about Abhi's project"])
    
    # Process
    ctx = hippo.process(ctx)
    
    print("Resulting Context Memories:")
    for mem in ctx.memories:
        print(f"- {mem}")
        
    print("Verification complete.")

if __name__ == "__main__":
    test_hippocampus()
