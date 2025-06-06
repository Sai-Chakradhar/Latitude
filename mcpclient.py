# mcp_streamlit_chat.py

import os
import asyncio
import streamlit as st
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("The OPENAI_API_KEY is missing in .env or system environment.")

# --- API Key Setup ---
os.environ["OPENAI_API_KEY"] = api_key

# --- MCP Server Path (point this to your actual server file) ---
SERVER_PATH = "/Users/saichakradhar/Desktop/Latittude/Server2.py"

# --- Streamlit Setup ---
st.set_page_config(page_title="🧠 MCP Agent Chat", layout="wide")
st.title("💬 Chat with Clinical Agent")

# Keep a running chat history in session_state
if "messages" not in st.session_state:
    st.session_state.messages = []


# Render whatever is already in st.session_state.messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])



async def run_agent(prompt: str):
    server_params = StdioServerParameters(command="python", args=[SERVER_PATH])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent("openai:gpt-4o", tools)

            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            history.append({"role": "user", "content": prompt})

            try:
                result = await agent.ainvoke({"messages": history})

                # Extract all messages
                tool_outputs = []
                final_ai_content = ""

                for m in result.get("messages", []):
                    if isinstance(m, ToolMessage):
                        tool_outputs.append(m.content)
                    elif isinstance(m, AIMessage):
                        final_ai_content = m.content  # The final response

                return tool_outputs, final_ai_content

            except Exception as e:
                import traceback
                traceback.print_exc()
                return [], f" Agent crashed: {e}"

# ⇒ Handle new user input
prompt = st.chat_input("Ask a clinical question or run a tool...")
if prompt:
    # 1) Display the user’s bubble immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Call the agent
    with st.spinner("Thinking..."):
        tool_outputs, ai_reply = asyncio.run(run_agent(prompt))
    for out in tool_outputs:
        st.session_state.messages.append({"role": "assistant", "content": out})
        with st.chat_message("assistant"):
            st.markdown(f"**Tool ▶**  {out}")

    st.session_state.messages.append({"role": "assistant", "content": ai_reply})
    with st.chat_message("assistant"):
        st.markdown(ai_reply)