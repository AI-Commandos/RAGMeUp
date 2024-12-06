
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


load_dotenv()
if __name__ == "__main__":
    db = SQLDatabase.from_uri("sqlite:///data/chinook.db")
    print(db.run("SELECT COUNT(*) FROM artists"))
    llm = ChatOllama(           
        model="llama3.2",
        temperature=0,
        # num_predict=40,
        # other params...
    )
    
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    system_message = prompt_template.format(dialect="SQLite", top_k=10)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_react_agent(
        llm, toolkit.get_tools(), state_modifier=system_message
    )
    # example_query = "What are the names of all the artists in the artists database?"
    # response = agent_executor.invoke({"messages": [("user", example_query)]})
    # print(response)
    tools = toolkit.get_tools()
    tools.append(multiply)
    chat_prompt = f"You are a helpfull assistant. Determine if you need to query the database or the multiply tool. If you need to use the SQL database, then do: {system_message}"
    chat_executor = create_react_agent(
        llm, tools = tools, state_modifier=chat_prompt
    )
    chat_query = "How many artist in artists table?"
    # chat_response = chat_executor.invoke({"messages": [("user", chat_query)]})
    # print(chat_response)
    events = chat_executor.stream(
        {"messages": [("user", chat_query)]},
        {"recursion_limit": 100},
        stream_mode="values",
    )
    for event in events:
        event["messages"][-1].pretty_print()