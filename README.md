# 🎓 GPTeach-by-example

This is an app that learns from your examples and provides answers based on relative examples by using OpenAI (LLM and embeddings), Pinecone (Vectorstore), and Langchain.

## How does it work?

1. Create a .csv file with two columns, first one named 'input' and second one named 'output'
2. Enter in as many examples in the .csv file as you'd like
3. Upload the .csv file to the app
4. The examples are converted into embeddings and stored in the vector store (Pinecone)
5. When submitting a question, langchain is used to embed the input and do a semantic similarity search of the vector store, which returns the top 3 results as context
6. The LLM generates a response using the 3 examples as context

## Get an OpenAI API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://platform.openai.com/account/api-keys.
2. Click on the `+ Create new secret key` button.
3. Next, enter an identifier name (optional) and click on the `Create secret key` button.

## Get a Pinecone API key

You can get your own OpenAI API key by following the following instructions:
1. Go to https://app.pinecone.io/.
2. Click on the `API Keys` button on the left panel sidebar.
3. Click `+ Create API Key` button in the top right.
3. Next, enter an identifier name and click on the `Create Key` button.
4. Take note of the Environment listed next to your API key.

## Try out the app

Once the app is loaded, go ahead and upload your examples, enter your API keys, and type a question in the text box and wait for a generated response.
