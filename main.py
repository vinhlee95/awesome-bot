from dotenv import load_dotenv
import os

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI()

prompt = ChatPromptTemplate(
  input_variables=["content"],
  messages=[
    HumanMessagePromptTemplate.from_template("{content}")
  ]
)

chain = LLMChain(
  llm=chat,
  prompt=prompt,
)

while True:
  content = input(">> ")

  result = chain({"content": content})

  print(result.get("text"))