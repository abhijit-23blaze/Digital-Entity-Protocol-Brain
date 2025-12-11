import argparse
import asyncio
import os
from dotenv import load_dotenv
from brain.network import BrainNetwork

# Load env variables
load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="DEP Agentic Brain CLI")
    parser.add_argument("query", type=str, help="The query to process")
    parser.add_argument("--flow", choices=["auto", "sequential", "parallel", "logical", "creative", "fast"], default="auto", help="The processing flow")
    
    args = parser.parse_args()
    
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set in .env or environment.")
        return

    print(f"Initializing DEP Brain [Mode: {args.flow.upper()}]...")
    
    network = BrainNetwork()
    
    try:
        if args.flow == "auto":
             result = await network.run_dynamic(args.query)
        elif args.flow == "fast":
            result = network.run_fast(args.query)
        elif args.flow == "sequential":
            result = network.run_sequential(args.query)
        elif args.flow == "logical":
            result = network.run_logical(args.query)
        elif args.flow == "creative":
            result = network.run_creative(args.query)
        else:
            result = await network.run_parallel(args.query)
            
        print("\n--- Final Output ---")
        print(result.final_output)
        
    except Exception as e:
        print(f"\nSystem Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
