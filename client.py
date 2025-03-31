import asyncio
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI

from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env


class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        self.client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),base_url=base_url)
    # methods will go here
    async def connect_to_server(self, server_script_path: str):
      """Connect to an MCP server

      Args:
          server_script_path: Path to the server script (.py or .js)
      """
      is_python = server_script_path.endswith('.py')
      is_js = server_script_path.endswith('.js')
      if not (is_python or is_js):
          raise ValueError("Server script must be a .py or .js file")

      command = "python" if is_python else "node"
      server_params = StdioServerParameters(
          command=command,
          args=[server_script_path],
          env=None
      )

      stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
      self.stdio, self.write = stdio_transport
      self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

      await self.session.initialize()

      # List available tools
      response = await self.session.list_tools()
      tools = response.tools
      print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
      """Process a query using Claude and available tools"""
      messages = [
          {
              "role": "user",
              "content": query
          }
      ]

      response = await self.session.list_tools()
      available_tools = [{
          "name": tool.name,
          "description": tool.description,
          "input_schema": tool.inputSchema
      } for tool in response.tools]

      tools = []
      for tool in response.tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema # Use 'inputSchema' as 'parameters'
            }
        })

      response = self.client.chat.completions.create(
          model="gemini-2.5-pro-exp-03-25",
          messages=messages,
          tools=tools
      )

      # Process response and handle tool calls
      final_text = []

      assistant_message_content = []
      message = response.choices[0].message
      #print(message)
      #print(message.content)
      print(message.tool_calls if message.tool_calls is not None else "No tool calls")
      if message.content is not None:
          final_text.append(message.content)
          assistant_message_content.append(message)
      import json
      from typing import Any
      if message.tool_calls is not None:
        for tool in message.tool_calls:
          tool_name: str = tool.function.name
          arg = tool.function.arguments
          tool_args: dict[str, Any] = json.loads(arg)

          # Execute tool call
          if tool_name is not None and tool_args is not None and self.session is not None:
            result = await self.session.call_tool(tool_name, tool_args)
            print("Assistant: ", result.content[0].text)
          final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

          assistant_message_content.append(message)
          messages.append({
              "role": "assistant",
              "content": assistant_message_content
          })
          messages.append(
              {
              "role": "user",
              "content": [
                  {
                      "type": "tool_result",
                      "content": result.content
                  }
              ]
          })

          response = self.client.chat.completions.create(
              model="gemini-2.5-pro-exp-03-25",
              messages=messages,
              tools=tools
          )

          final_text.append(response.choices[0].message.content)

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
                print(f"\nError: {str(e)}")

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
