# from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from dotenv import load_dotenv
import os
import streamlit as st
import json
from langchain_openai import ChatOpenAI
# from langchain.agents import AgentType

load_dotenv()
API_KEY=os.getenv("OPENAI_API_KEY")

def agent(filename: str):

    llm = ChatOpenAI(
        model = "gpt-3.5-turbo",
        temperature = 0.0,
        # max_tokens = 256,
        # top_p = 0.5,
    )
    df = pd.read_csv(filename, encoding='unicode_escape')

    pandas_df_agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=True, 
        # return_intermediate_steps=True,
        # agent_type=AgentType.OPENAI_FUNCTIONS
        )
    
    return pandas_df_agent

def get_response(agent, query):
    prompt = (
        """
            For the following query, if it requires drawing a table, reply as follows:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

            If the query requires creating a bar chart, reply as follows:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            If the query requires creating a line chart, reply as follows:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

            There can only be two types of charts, "bar" and "line".

            If it is just asking a question that requires neither, reply as follows:
            {"answer": "answer"}
            Example:
            {"answer": "The product with the highest sales is 'Classic Cars.'"}

            Write supportive numbers if there are any in the answer.
            Example:
            {"answer": "The product with the highest sales is 'Classic Cars' with 1111 sales."}

            If you do not know the answer, reply as follows:
            {"answer": "I do not know."}

            Do not hallucinate or make up data. If the data is not available, reply "I do not know."
            
            Return all output as a string in double quotes. 

            All strings in "columns" list and data list, should be in double quotes,

            For example: {"columns": ["title", "ratings_count"], "data": [["Gilead", 361], ["Spider's Web", 5164]]}

            Lets think step by step.

            Below is the query.
            Query: 
            """
        + query
    )

#             If the query requires creating a scatter plot, reply as follows:
#             {"scatter": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

#             If the query requires creating a histogram, reply as follows:
#             {"histogram": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

#             There can only be four types of chart, "bar", "line", "scatter" and "histogram".

    response = agent.run(prompt)
    return response.__str__()

def return_response(response: str) -> dict: 
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        return None

def write_response(response_dict: dict):
    if response_dict is not None:
        if "answer" in response_dict:
            answer = response_dict["answer"]
            st.write(answer)

        if "bar" in response_dict:
            data = response_dict["bar"]
            df = pd.DataFrame.from_dict(data, orient = 'index')
            df = df.transpose()
            df.set_index("columns", inplace=True)
            st.bar_chart(df)

        if "line" in response_dict:
            data = response_dict["line"]
            df = pd.DataFrame(data)
            df.set_index("columns", inplace=True)
            st.line_chart(df)

        # if "scatter" in response_dict:
        #     data = response_dict["scatter"]
        #     df = pd.DataFrame(data)
        #     df.set_index("columns", inplace=True)
        #     st.scatter_chart(df)

        # if "histogram" in response_dict:
        #     data = response_dict["histogram"]
        #     df = pd.DataFrame(data)
        #     df.set_index("columns", inplace=True)
        #     st.plotly_chart(df)

        if "table" in response_dict:
            data = response_dict["table"]
            df = pd.DataFrame(data["data"], columns=data["columns"])
            st.table(df)

    else:
        st.write("Decoded response is None. Please retry with a better prompt.")
