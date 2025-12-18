import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv

load_dotenv()

df = pd.read_csv("employee.csv")
print(df.head())



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,
    allow_dangerous_code=True  # needed for calculations
)

#query = "What is the average salary?"
query = "What is the highest salary?"
response = agent.invoke(query)
'''
questions = [
    "How many employees are there?",
    "List employees from Bangalore",
    "What is the highest salary?",
    "Show top 3 employees by experience",
    "Average salary by department"
]

for q in questions:
    print(f"\nQuestion: {q}")
    print(agent.invoke(q)["output"])'''

print(response["output"])
