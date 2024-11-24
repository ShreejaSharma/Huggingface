pip install -qq langchain wget llama-index cohere llama-cpp-python
import wget
model_url = 'https://huggingface.co/TheBloke/Llama-2-7B-GGUF'
def bar_custom(current, total, width=80):
  print("Downloading")
wget.download(model_url,bar=bar_custom)

pip q install streamlit
%%writefile app.py
import streamlit as st
from llama-index import (SimpleDirectoryReader, VectorStoreIndex, ServiceContext)
from llama-index.llms import LlamaCPP
from llama-index.llms.llama_utils import (messages_to_prompt, completion_to_prompt)
from langchain.schema import(HumanMessage,AIMessage)

def __init__page() -> None:
  st.set_page_config(page_title="Personal Chatbot")
  st.header("Personal Chatbot")
  st.sidebar.title("Options")

def select__llm() -> LlamaCPP:
  return LlamaCPP(model_path="llama-2-7b-chat.Q4_K_M.gguf",temperature=0.1,max_new_tokens=500,context_window=3900,model_kwargs={"n_gpu_layers":1},messages_to_prompts=messages_to_prompts,verbose=True,generate_kwargs={})
  
 def init_messages() -> None:
   clear_button = st.sidebar.button("Clear Conversation")
   if clear_button or "messages" not in st.session_state:
   st.session_state.messages = [SystemMessage(content="you are an helpful AI assistant. Reply your answer in markdown format.")]

 def generate_answer(llm, messages) -> str:
   response = llm.complete(message)
   return response.text


def main():
  init_page()
  llm = select_llm()
  init_message()
  if user_input := st.chat_input("Input your question!"):
    st.session_state.messages.append(HumanMessage(content=user_input))
    with st.spinner("Bot is typing..."):
      answer = get_answer(llm,user_input)
    st.session_state.messages(AImessage(content=answer))
  
  
  messages = st.session_state.get("messages", [])
  for message in messages:
    if isInstance(message, AImessage):
      with st.chat_message("assistant"):
        st.markdown(message.content)
    elif isInstance(message, Humanmessage):
      with st.chat_message("user"):
        st.markdown(message.content)


  if __name__=="__main__":
    main()
    
  
   
