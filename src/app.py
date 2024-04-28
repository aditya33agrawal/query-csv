import streamlit as st
from utils import get_response, agent, return_response, write_response

st.write("Upload your CSV file below: ")

data = st.file_uploader("Upload the CSV", type=['csv'])

question = st.text_area("Type in your question")

if st.button("Submit Question", type="primary"):
    csv_agent = agent(data)
    response = get_response(agent = csv_agent, query = question)
    decoded_response = return_response(response)
    write_response(decoded_response)

