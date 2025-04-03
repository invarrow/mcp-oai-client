import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from dotenv import load_dotenv
import os
import json

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self, config_file="mcp_config.json"):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        self.client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url=base_url)
        self.model="gemini-2.0-flash"
        self.messages = []
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.mcp_servers = config.get("mcpServers", {})

    async def connect_to_server(self, server_name: str):
        if server_name not in self.mcp_servers:
            raise ValueError(f"Server '{server_name}' not found in config.")

        server_config = self.mcp_servers[server_name]
        command = server_config.get("command")
        args = server_config.get("args", [])
        env = server_config.get("env", {})

        server_params = StdioServerParameters(command=command, args=args, env=env)
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
      """Process a query using Claude and available tools"""
      self.messages.append({
          "role": "user",
          "content": query
          }
      )

      response = await self.session.list_tools()
      tools = []
      for i,tool in enumerate(response.tools):
        if i==2:continue
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema # Use 'inputSchema' as 'parameters'
            }
        })

      response = self.client.chat.completions.create(
          model=self.model,
          messages=self.messages,
          tools=tools
      )

      # Process response and handle tool calls
      final_text = []

      message = response.choices[0].message
      if message.content is not None:
          final_text.append(message.content)
      import json
      from typing import Any
      if message.tool_calls is not None:
        for tool in message.tool_calls:
          tool_name: str = tool.function.name
          arg = tool.function.arguments
          tool_args: dict[str, Any] = json.loads(arg)

          # Execute tool call
          if tool_name is not None and tool_args is not None and self.session is not None:
            final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
            result = await self.session.call_tool(tool_name, tool_args)
            print("Tool : ", result.content[0])
            self.messages.append({
                "role": "assistant",
                "content": f"Data returned from tool call {tool_name} with args {tool_args}\n"+str(result.content[0])
            })
            self.messages.append({"role": "user","content": "Explain the above data from the tool call {tool_name} with args {tool_args} based on the {query}"})
          response = self.client.chat.completions.create(
              model=self.model,
              messages=self.messages,
              tools=tools
          )
          try:
            if response.choices: final_text.append(response.choices[0].message.content)
            print("Assistant: ",response.choices[0])
          except Exception as e:
            print("Error: ", e)


      return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                print("\n" + response)

            except Exception as e:
                print(f"\nErrorr: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
