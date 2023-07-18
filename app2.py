import time
import gradio as gr
import os
import re
import fitz
import numpy as np
import openai
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors
import logging

recommender = None
chunk_len = 400
overlap = 20
model="text-embedding-ada-002"
def pdf_to_chunks(path):
    text_list = []
    with fitz.open(path) as doc:
        total_pages = doc.page_count
        for i in range(total_pages):
            text = doc.load_page(i).get_text("text")
            text = text.replace('\n', ' ')
            text = re.sub('\s+', ' ', text)
            text_list.append(text)

    page_nums = []
    chunks = []
    text_toks = [t.split(' ') for t in text_list]
    for idx, words in enumerate(text_toks):
        for i in range(0, len(words), chunk_len-overlap):
            chunk = words[i : i + chunk_len]
            if (
                (i + chunk_len) > len(words)
                and (len(chunk) < chunk_len)
                and (len(text_toks) != (idx + 1))
            ):
                text_toks[idx + 1] = chunk + text_toks[idx + 1]
                continue
            chunk = ' '.join(chunk).strip()
            chunk = f'[Page no. {idx}]' + ' ' + '"' + chunk + '"'
            chunks.append(chunk)
    return chunks

class SemanticSearch:
    def __init__(self):
        self.fitted = False

    def fit(self, data, n_neighbors=2):
        self.data = data
        self.embeddings = self.get_text_embedding(data)
        n_neighbors = min(n_neighbors, len(self.embeddings))
        self.nn = NearestNeighbors(n_neighbors=n_neighbors)
        self.nn.fit(self.embeddings)
        self.fitted = True

    def __call__(self, text, return_data=True):
        inp_emb = openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
        inp_emb=[inp_emb]
        neighbors = self.nn.kneighbors(inp_emb, return_distance=False)[0]

        if return_data:
            return [self.data[i] for i in neighbors]
        else:
            return neighbors

    def get_text_embedding(self, texts):
        embeddings = []
        openai.api_key=os.environ["OPENAI_API_KEY"]
        for i in range(0, len(texts)):
            emb_batch = openai.Embedding.create(input = texts[i], model=model)['data'][0]['embedding']
            embeddings.append(emb_batch)
        embeddings = np.vstack(embeddings)
        return embeddings

def loading_pdf():
    return "Loading..."

def pdf_changes(selected, openai_key):
    path = select(selected)
    global recommender 
    recommender = None
    if openai_key is not None:
        os.environ['OPENAI_API_KEY'] = openai_key
        chunks = pdf_to_chunks(path)
        if recommender is None:
            recommender = SemanticSearch()
            recommender.fit(chunks)
        return "Ready"
    else:
        return "You forgot OpenAI API key"
    
system_message = {"role": "system", "content": "You are a helpful assistant."}
def select(selected):
  return selected +'.pdf'


css="""
#col-container {max-width: 700px; margin-left: auto; margin-right: auto;}
"""
title = """
<div style="text-align: center;max-width: 1000px;">
    <h1>Chat with PDF • OpenAI</h1>
    <p style="text-align: center;">Upload a .PDF from your computer, click the "Load PDF to LangChain" button, <br />
    when everything is ready, you can start asking questions about the pdf ;) <br />
    This version is set to store chat history, and uses OpenAI as LLM, don't forget to copy/paste your OpenAI API key</p>
</div>
"""
options = ['model', 'meditation'] 

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(title)   
        with gr.Column():
            openai_key = gr.Textbox(label="You OpenAI API key", type="password")
            selected = gr.inputs.Dropdown(options,label= "Choose the interview transcript", default = options[1])
            with gr.Row():
                system_status = gr.Textbox(label="Status", placeholder="", interactive=False)
                load_pdf = gr.Button("Load the interview")
            load_pdf.click(loading_pdf, None, system_status, queue=False)    
            load_pdf.click(pdf_changes, inputs=[selected, openai_key], outputs=[system_status], queue=False)

            chatbot = gr.Chatbot()
            question = gr.Textbox(show_label=False, placeholder="Ask me a question and press enter.").style(container=False)
            clear = gr.Button("Clear")
            state = gr.State([])

            def user(user_message, history):
                return "", history + [[user_message, None]]

            def bot(history, messages_history):
                user_message = history[-1][0]
                bot_message, messages_history = ask_gpt(user_message, messages_history)
                messages_history += [{"role": "assistant", "content": bot_message}]
                history[-1][1] = bot_message
                time.sleep(1)
                return history, messages_history

            def ask_gpt(question, messages_history):
                openai.api_key=os.environ["OPENAI_API_KEY"]
                topn_chunks = recommender(question)
                prompt = ""
                prompt += 'search results:\n\n'
                for c in topn_chunks:
                    prompt += c + '\n'
                prompt += "Instructions: Compose a comprehensive reply to the query\
                based on the messages and serach results given. The answer should be\
                accurate, short and concise. Answer step-by-step. \nQuery: {question}\nAnswer: "
                

                prompt += f"Query: {question}\nAnswer:"
                if(len(messages_history)>2):
                    messages_history.pop(1)
                messages_history += [{"role": "user", "content": prompt}]
                #my_history = messages_history + [{"role": "user", "content": prompt}]
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=messages_history,
                    max_tokens = 200
                )
                return response['choices'][0]['message']['content'], messages_history

            def init_history(messages_history):
                messages_history = []
                messages_history += [system_message]
                return messages_history

            question.submit(user, [question, chatbot], [question, chatbot], queue=False).then(
                bot, [chatbot, state], [chatbot, state]
            )

            clear.click(lambda: None, None, chatbot, queue=False).success(init_history, [state], [state])


demo.launch() # `enable_queue=True` to ensure the validity of multi-user requests


