import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from mcp_use import MCPClient, MCPAgent


import os

async def run_memory_chat():

    load_dotenv()

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    config_file = "browser_mcp.json"

    print("initializing chat ....")

    client = MCPClient.from_config_file(config_file)
    llm = ChatGroq(model_name="qwen-qwq-32b")

    agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=15,
        memory_enabled=True,
        )

    print("\n===== Interactive MCP chat =====")
    print("type 'exit' or 'quit' to end the chat")
    print(" type 'clear' to clear conversation history")
    print("=================================\n")


    try:
        while True:
            user_input = input("\nYou: ")

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting chat...")
                break

            if user_input.lower() == "clear":
                agent.clear_conversation_history()
                print("Conversation history cleared.")
                continue

            print("\n Assistant: ", end="", flush=True)

            try:
                response = await agent.run(user_input)
                print(response)

            except Exception as e:
                print(f"\n error: {e}")

    finally:

        if client and client.sessions:
            await client.close_all_sessions()

if __name__ == "__main__":
    asyncio.run(run_memory_chat())
            
            

    
    
    
