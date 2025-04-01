import os
import dotenv
import streamlit as st
import requests
from transformers import AutoTokenizer

dotenv.load_dotenv()
token = os.environ['HF_TOKEN']

models = {
    'google/gemma-3-12b-it' : '[/INST]',
    'google/gemma-3-27b-it': '<start_of_turn>model\n',
}

model = st.selectbox('Select desired model:', options=models )
token_model = models[model]


if 'current_model ' not in st.session_state or st.session_state['current_model'] != model:
    st.session_state['current_model'] = model
    st.session_state['messages'] = [ ]

model_name = st.session_state['current_model']

tokenizer = AutoTokenizer.from_pretrained(model_name)
url = f'https://api-inference.huggingface.co/models/{model_name}'
messages = st.session_state['messages']

area_chat = st.empty()

user_input = st.chat_input('Ask away: ')
if user_input:
    messages.append({'role' : 'user', 'content': user_input})
    template = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    json = {
        'inputs': template,
        'parameters': {'max_new_tokens': 1000},
        'options': {'use_cache': False, 'wait_for_model': True},
    }

    headers = {'Authorization': f'Bearer {token}'}

    response = requests.post(url, json=json, headers=headers).json()
    chatbot_message = response[0]['generated_text'].split(token_model)[-1]
    messages.append({'role':'assistant', 'content': chatbot_message })
     
with area_chat.container():
    for message in messages:
        chat = st.chat_message(message['role'])
        chat.markdown(message['content'])
