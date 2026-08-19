import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    def add_markdown(source: str):
        notebook["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True)
        })

    def add_code(source: str):
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True)
        })

    theory_md = """# Day 31: Implement SQL RAG

Welcome to Day 31! Today we are tackling **SQL RAG** (Retrieval-Augmented Generation on Structured Data). We will use LangChain to translate natural language queries into executable SQL against a SQLite database.

## Core Theory (Just-in-Time)

### The "Why"
Most enterprise data doesn't live in PDFs or unstructured text; it lives in relational databases. Traditional RAG (chunking text and using vector search) fails miserably on structured, tabular data. You can't run a `SUM()`, `GROUP BY`, or `JOIN` using a vector similarity search. To unlock insights from relational databases, we need the LLM to act as a translator—turning a user's natural language question into a syntactically correct SQL query.

### The "How"
The architecture for SQL RAG typically follows these steps:
1.  **Schema Retrieval:** The LLM is provided with the database dialect and the schema (table names, column names, data types, and perhaps sample rows) of the relevant tables.
2.  **Query Generation:** The LLM generates a SQL query based on the user's question and the retrieved schema.
3.  **Query Execution:** The generated SQL is executed safely against the database.
4.  **Answer Synthesis (Optional):** The results of the query are fed back into the LLM to generate a natural language response summarizing the findings.

We will use LangChain's standard tools (specifically `create_sql_query_chain`) to manage this workflow.

## Common Pitfalls in Production
1.  **Security Risks (SQL Injection):** Never give the LLM an LLM connection with write/drop permissions. **Always use read-only credentials.** If the LLM hallucinates a `DROP TABLE`, it should fail at the database permission level.
2.  **Schema Complexity:** Feeding an entire enterprise database schema (with 500+ tables) into an LLM context window will fail. You must filter or retrieve only the relevant table schemas *before* generation.
3.  **Hallucinated Columns:** LLMs often guess column names. Providing a few sample rows alongside the schema significantly reduces this.
4.  **Dialect Mismatch:** Forgetting to tell the LLM which SQL dialect to use (e.g., PostgreSQL vs. SQLite) leads to syntax errors in generated queries."""

    add_markdown(theory_md)

    setup_md = """## Setup and Dependencies

Let's install the necessary packages. We'll use `langchain`, `langchain-community`, `langchain-openai`, and `sqlalchemy`. We will use SQLite as our database."""

    add_markdown(setup_md)

    setup_code = """# Run this in your terminal if you haven't already:
# uv pip install langchain langchain-openai langchain-community sqlalchemy"""

    add_code(setup_code)

    db_setup_md = """## 1. Database Setup

First, let's create a sample SQLite database using `sqlite3` and `sqlalchemy` to act as our data source."""

    add_markdown(db_setup_md)

    db_setup_code = """import sqlite3
from typing import List, Tuple

def setup_sample_db(db_path: str = "ecommerce.db") -> None:
    \"\"\"Creates a sample SQLite database with users and orders tables.\"\"\"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            signup_date DATE
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY,
            user_id INTEGER,
            product_name TEXT NOT NULL,
            amount REAL NOT NULL,
            order_date DATE,
            FOREIGN KEY(user_id) REFERENCES users(user_id)
        )
    ''')

    # Insert sample data
    cursor.execute("DELETE FROM users")
    cursor.execute("DELETE FROM orders")

    users: List[Tuple[int, str, str, str]] = [
        (1, 'Alice Smith', 'alice@example.com', '2023-01-15'),
        (2, 'Bob Johnson', 'bob@example.com', '2023-02-20'),
        (3, 'Charlie Brown', 'charlie@example.com', '2023-03-10')
    ]
    cursor.executemany("INSERT INTO users VALUES (?, ?, ?, ?)", users)

    orders: List[Tuple[int, int, str, float, str]] = [
        (101, 1, 'Laptop', 1200.50, '2023-04-01'),
        (102, 1, 'Mouse', 25.00, '2023-04-02'),
        (103, 2, 'Monitor', 300.00, '2023-04-10'),
        (104, 3, 'Keyboard', 75.00, '2023-04-15'),
        (105, 1, 'Headphones', 150.00, '2023-04-20')
    ]
    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    conn.commit()
    conn.close()
    print(f"Sample database initialized at {db_path}")

setup_sample_db()"""

    add_code(db_setup_code)

    sql_rag_md = """## 2. Implementing SQL RAG with LangChain

Now we will connect LangChain to our database and build a chain that translates a natural language question into a SQL query."""

    add_markdown(sql_rag_md)

    sql_rag_code = """import os
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Ensure you have your OpenAI API key set.
# os.environ["OPENAI_API_KEY"] = "your-api-key"

# 1. Connect to the database using SQLAlchemy (via LangChain's wrapper)
db = SQLDatabase.from_uri("sqlite:///ecommerce.db")
print("Dialect:", db.dialect)
print("Usable tables:", db.get_usable_table_names())

# Initialize the LLM
# We use a chat model with temperature 0 for deterministic SQL generation
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. Create the SQL Query Chain
# This built-in chain inspects the db schema, the dialect, and creates a prompt
# asking the LLM to write a SQL query.
generate_query_chain = create_sql_query_chain(llm, db)

def extract_sql(text: str) -> str:
    \"\"\"Cleans up the SQL string returned by the LLM by removing markdown formatting.\"\"\"
    return text.replace("```sql", "").replace("```", "").strip()

def run_sql_rag(question: str) -> str:
    \"\"\"
    End-to-end function that takes a natural language question,
    generates a SQL query, executes it, and returns the result.
    \"\"\"
    print(f"\\n--- Question: {question} ---")

    # Step A: Generate the query
    generated_sql_raw = generate_query_chain.invoke({"question": question})
    clean_sql = extract_sql(generated_sql_raw)
    print(f"Generated SQL: \\n{clean_sql}\\n")

    # Step B: Execute the query securely against the DB
    try:
        # Warning: In a real production system, use read-only credentials!
        result = db.run(clean_sql)
        print(f"Execution Result: {result}")
        return result
    except Exception as e:
        print(f"Error executing query: {e}")
        return str(e)

# Test it out!
run_sql_rag("How many users are in the database?")
run_sql_rag("What is the total amount spent by Alice Smith?")
run_sql_rag("Which product is the most expensive?")"""

    add_code(sql_rag_code)

    lab_md = """## 3. Practical Lab / Homework

Your task is to extend the basic SQL RAG implementation to include an **Answer Synthesis** step.

Currently, our function prints raw database results like `[(1375.5,)]`. We want the LLM to take that raw result and the original question, and form a polite, natural language response.

**Task:**
1.  Write a new prompt template that accepts three variables: `question`, `sql_query`, and `sql_result`.
2.  Create a LangChain LCEL (LangChain Expression Language) pipeline that:
    - Passes the original question through.
    - Generates the SQL query.
    - Executes the SQL query using `db.run()`. (Hint: You can use `RunnableLambda` to wrap standard python functions in LCEL, or use LangChain's built-in `QuerySQLDataBaseTool`).
    - Feeds the question, the query, and the execution results into your new prompt.
    - Uses the LLM to generate the final natural language answer.
3.  Test your new full pipeline with the question: `"What is the average order amount?"`"""

    add_markdown(lab_md)

    lab_code = """from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough

# Your implementation here

# 1. Define the synthesis prompt
answer_prompt = PromptTemplate.from_template(
    \"\"\"Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: \"\"\"
)

# 2. Set up the components
execute_query_tool = QuerySQLDataBaseTool(db=db)
write_query_chain = create_sql_query_chain(llm, db) | extract_sql

# 3. Build the full LCEL chain
# The chain should:
# a) Generate the query
# b) Execute the query
# c) Pass everything to the answer_prompt
# d) Generate the final answer using the llm

full_chain = (
    RunnablePassthrough.assign(query=write_query_chain)
    .assign(result=lambda inputs: execute_query_tool.invoke({"query": inputs["query"]}))
    | answer_prompt
    | llm
    | StrOutputParser()
)

# 4. Test the chain
# response = full_chain.invoke({"question": "What is the average order amount?"})
# print(response)
"""

    add_code(lab_code)

    with open("modules/day_31.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

if __name__ == "__main__":
    create_notebook()
    print("Notebook modules/day_31.ipynb created successfully.")
