from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-20b", streaming=True)
search = GoogleSerperAPIWrapper()
tool = [search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []

agent = create_agent(
    model=llm,
    tools=tool,
    system_prompt="you are an ai agent that can search on google as well.",
    checkpointer=st.session_state.memory
)

st.title("🤖 Agentic Chatbot")
st.subheader("Welcome in Faster Google Search ai agent Chatbot..")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything? ")

if query:
    st.session_state.history.append({"role" : "user", "content" : query})
    st.chat_message("user").markdown(query)
    response = agent.stream(
            {"messages" : [{"role" : "user", "content" : query}]},
            {"configurable" : {"thread_id" : "1"}},
            stream_mode = "messages"
        )
    AI_Container = st.chat_message("ai")
    with AI_Container:
        space = st.empty()
        message = ""
        
        for chunk in response:
            message = message + chunk[0].content
            space.write(message)
        
        st.session_state.history.append({"role" : "ai", "content" : message})
