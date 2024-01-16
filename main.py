from dotenv import load_dotenv
import os

from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
# from langchain_openai import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, FileChatMessageHistory, PostgresChatMessageHistory

# Load env file having OPENAI_API_KEY variable
load_dotenv()
chat = ChatOpenAI(verbose=True)

"""
- memory_key: the key that will be used to store the memory in the memory object. 
              It needs to be similar to the input variable name in the prompt template (ChatPromptTemplate).
- returnMessages: true makes the memory return a list of chat messages instead of a string.
"""
chat_history_db = PostgresChatMessageHistory(
  session_id="1",
  connection_string="postgresql://@localhost:5432/awesome-bot",
  table_name="chat_history"
)

"""
Store conversation history in a local file or database. 
When the main chain runs, it inject the past conversation into the prompt as an input variable. 
"""
# memory = ConversationBufferMemory(
#   chat_memory=chat_history_db,
#   # Enable this for a local file chat history
#   # chat_memory=FileChatMessageHistory("chat_history.json"),
#   memory_key="messages", 
#   return_messages=True
# )

"""
Alternative type of memory that:
- summarise the past conversations over time and store in memory
- when the main chain runs, it inject the summarised conversation into the prompt as SystemMessage
- this memory is most useful for longer conversations, 
  where keeping the past message history in the prompt verbatim would take up too many tokens.

https://python.langchain.com/docs/modules/memory/types/summary
"""
memory = ConversationSummaryMemory(
  memory_key="messages",
  llm=chat,
  return_messages=True,
)

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