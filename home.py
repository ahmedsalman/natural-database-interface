import streamlit as st

st.set_page_config(
    page_title="Home",
    page_icon="🏠",
)

st.markdown(
    """
# ChatDB
A web app built with [Streamlit](https://streamlit.io/) that allows you to chat with your databases using a GPT model.

## Getting started
Start by adding some database connections in the *⚙️ Settings* page.

- Go to the **⚙️ Settings** page
- Provide your OpenAI API key and add some database connnections
- Go to the **🤖 Chats** page
- Create a new conversation by specifying the GPT model and selecting the databases you added
- Start chatting 🗣 ↔️ 🤖

## Compatibility
Currently supported databases:
- PostgreSQL (using `psycopg2`)
- MySQL and MariaDB (using `mysqlclient`)
- Oracle (using `cx_oracle`)
- Microsoft SQL Server (using `pyodbc`)
"""
)
