# loading the libraries and local packages
import os
from langchain.vectorstores import Chroma
from src.helper import load_embedding, repo_ingestion
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain

# initialize the flask application
app = Flask(__name__)


# Step-01 : Setting the API KEY for embedding
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
 
# Step-02 : calling load_embedding() function
embeddings = load_embedding()

# assigning varibale
persist_directory = "db"

# Step-03 : storing data into vector store here using chromadb
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
    )


# Step-04 : model loading through ChatOpenAI() class
llm = ChatOpenAI()


# Step-05 : conversation buffer memory
memory = ConversationSummaryMemory(llm=llm, memory_key = "chat_history", return_messages=True)


# Step-06 : loading ConversationalRetrievalChain clas
qa = ConversationalRetrievalChain.from_llm(llm, retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k":8}), memory=memory)




@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html')



@app.route('/chatbot', methods=["GET", "POST"])
def gitRepo():

    if request.method == 'POST':
        user_input = request.form['question']
        repo_ingestion(user_input)
        os.system("python store_index.py")

    return jsonify({"response": str(user_input) })



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)

    if input == "clear":
        os.system("rm -rf repo")

    result = qa(input)
    print(result['answer'])
    return str(result["answer"])





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)