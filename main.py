from dotenv import load_dotenv
import os

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")

chat = ChatOpenAI()

"""
- memory_key: the key that will be used to store the memory in the memory object. 
              It needs to be similar to the input variable name in the prompt template (ChatPromptTemplate).
- returnMessages: true makes the memory return a list of chat messages instead of a string.
"""
memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

prompt = ChatPromptTemplate(
  # This property specifies the input variables that will be passed to the prompt.
  # In this case we have 2:
  # - Content: what the user says to the chatbot
  # - Messages: the chat history, for the chatbot to be conversational
  input_variables=["content", "messages"],
  messages=[
    HumanMessagePromptTemplate.from_template("{content}"),
    MessagesPlaceholder("messages"),
  ]
)

# Form the main chain
chain = LLMChain(
  llm=chat,
  prompt=prompt,
  # https://python.langchain.com/docs/modules/memory/adding_memory
  memory=memory,
  # Enable this option for better visibility on different phases of the chain (Entering, Finished)
  verbose=True
)

while True:
  content = input(">> ")

  result = chain({"content": content})

  print(result.get("text"))