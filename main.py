import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from streamlit_chat import message
import os
from dotenv import load_dotenv

load_dotenv()

def generate_response(uploaded_file, openai_api_key, pinecone_api_key, pinecone_env, index_name, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        # get field names/column header names
        column_names = df.columns.tolist()
        # Convert DataFrame to dictionary
        examples = df.to_dict(orient='records')
        example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Input: {input}\nOutput: {output}",
        )
        # Set embedding model
        model="text-embedding-ada-002"
        embed_model=OpenAIEmbeddings(model=model,openai_api_key=openai_api_key)
        # Initiate vector store
        import pinecone
        pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_env,
        )
        # Check if index already exists
        if index_name not in pinecone.list_indexes():
            # if does not exist, create index
            pinecone.create_index(
                index_name,
                dimension=1536,
                metric='cosine',
            )
        # Set up semantic example selector
        example_selector = SemanticSimilarityExampleSelector.from_examples(
        # These are the examples it has available to choose from.
        examples=examples,
        # This is the PromptTemplate being used to format the examples.
        example_prompt=example_prompt,
        # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
        embeddings=embed_model,
        index_name=index_name,
        # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
        vectorstore_cls=Pinecone,
        k=3,
        )
        dynamic_prompt = FewShotPromptTemplate(
        # We provide an ExampleSelector instead of examples.
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="Give a relative answer for every input. Do not try to make up an answer if you do not find relavent examples. Simply say that you could not find any relative examples to answer the question.",
        suffix="Input: {query}\nOutput: ", 
        input_variables=["query"],
        )
        # Define LLM for generating text
        import openai
        openai = OpenAI(
        model_name="text-davinci-003",
        openai_api_key=openai_api_key,
        temperature=0,
        )
        return openai(dynamic_prompt.format(query=query_text))

# Page title
st.set_page_config(page_title='🎓 GPTeach-by-example')
st.title('🎓 GPTeach-by-example')

# CSV file upload
uploaded_file = st.file_uploader('Upload a CSV file with two columns named input and output', type='csv')
# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Enter a question related to the provided examples', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    if os.getenv("OPENAI_API_KEY") is None or os.getenv("OPENAI_API_KEY") == "":
        openai_api_key = st.text_input('OpenAI API Key (https://platform.openai.com/)', type='password')
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    if os.getenv("PINECONE_API_KEY") is None or os.getenv("PINECONE_API_KEY") == "":    
        pinecone_api_key = st.text_input('Pinecone API Key (https://app.pinecone.io/)', type='password')
    else:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if os.getenv("PINECONE_ENV") is None or os.getenv("PINECONE_ENV") == "":     
        pinecone_env = st.text_input('Pinecone Environment')
    else:
        pinecone_env = os.getenv("PINECONE_ENV")
    if os.getenv("PINECONE_ENV") is None or os.getenv("PINECONE_ENV") == "":  
        index_name = st.text_input('Pinecone Index Name')
    else:
        index_name = os.getenv("INDEX_NAME")
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Looking up relavent examples... this may take a couple minutes if creating a new vector store index.'):
            response = generate_response(uploaded_file, openai_api_key, pinecone_api_key, pinecone_env, index_name, query_text)
            result.append(response)
            del openai_api_key
            del pinecone_api_key
            del pinecone_env
            del index_name

if len(result):
    st.info(result[0])
