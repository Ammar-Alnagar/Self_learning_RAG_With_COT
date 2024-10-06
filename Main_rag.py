from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
import json
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from Indexer import  index_conversations

    
  
# Load environment variables (such as API keys)
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API")

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=False)

# Create the Chroma vector store
db = Chroma(persist_directory="./db-mawared",
            embedding_function=embeddings)

# Create retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}  # Retrieve the top 5 most similar documents
)

# Initialize the LLM (Groq's version of LLaMA)
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.3,        # Deterministic output
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# Create a prompt template
template = """
You are an expert assistant specializing in the Mawared HR System. Your role is to answer user questions by systematically reasoning through the provided context. Use Chain-of-Thought (CoT) reasoning to break down and solve complex questions. If the context is insufficient, ask clarifying questions to gather more information.

Guidelines:

Use CoT Reasoning to break down the problem step-by-step.
Refer only to the provided context for information.
Be concise and direct in your final responses, but transparent in showing your reasoning process.
If context is insufficient, ask relevant follow-up questions to gather more information instead of speculating.
Refine your response after analyzing each step, and verify all details.
When responding to a question, follow these steps:

Analyze the Question:

Read the question carefully. Understand the details and any potential ambiguity.
Break it down into specific sub-questions or tasks if necessary.
Identify key elements and areas that may need clarification or further exploration.
State your assumptions explicitly if the context is incomplete.
Chain-of-Thought Reasoning:

Start by stating the approach you plan to take to answer the question.
List out intermediate steps that lead to the solution, ensuring each builds logically from the previous one.
Think out loud by asking yourself questions that explore how each step connects to the context.
Formulate Response:

Use contextual information to support each step of the reasoning.
Develop a clear and logical explanation based on your analysis.
Incorporate analogies or examples when appropriate to explain abstract concepts.
Verify and Refine:

After developing an answer, review it step-by-step to check for consistency and correctness.
Ensure that your response directly addresses the user's question and that all key points are supported by context.
Identify gaps that need clarification or further detail from the user.
Present the Answer:

Provide a clear, step-by-step response when appropriate, showing your reasoning process (CoT).
Use an engaging and accessible tone to ensure clarity.
Acknowledge any limitations due to incomplete context.
Ask follow-up questions if more information is needed.
Context: {context}

Question:
{question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Function to save conversation history
def save_conversation(question, answer, file_path='Data/conversation_history.json'):
    try:
        with open(file_path, 'r') as file:
            conversation_history = json.load(file)
    except FileNotFoundError:
        # If no history file is found, create a new one
        conversation_history = []

    # Append the current question-answer pair to the history
    conversation_history.append({"question": question, "answer": answer})

    # Save the updated conversation history to the file
    with open(file_path, 'w') as file:
        json.dump(conversation_history, file, indent=4)

# Function to ask questions and stream answers
def ask_question(question):
    print("Answer:\t", end=" ", flush=True)
    answer = ""
    
    # Stream the output chunk by chunk
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
        answer += chunk
    
    print("\n")  # Move to a new line after the response
    # Save the conversation
    save_conversation(question, answer)
    return answer

# Function to load conversation history (if needed for future context)
def load_conversation(file_path='Data/conversation_history.json'):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        # If no history file exists, return an empty list
        return []

# Example usage
if __name__ == "__main__":
    print("Welcome to the Mawared HR RAG system. Type 'quit' to exit.")
    
    # Run the indexer to index past conversation history
    # index_conversations()  # Call the index_conversations function

    # Load past conversation history
    conversation_history = load_conversation()

    while True:
        user_question = input("Ask a question (or type 'quit' to exit): ")
        if user_question.lower() == 'quit':
            # index_conversations()
            print("Exiting the system. Goodbye!")
            break
        
        # Ask the question and get a streamed response
        ask_question(user_question)
