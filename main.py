import pinecone
from dotenv import load_dotenv
import os
import base64

load_dotenv()

# Initialize Pinecone with API key
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create or connect to an index
index_name = "github-repo-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Change this to match your embeddings model's dimension
        metric="cosine",  # Use cosine distance for similarity search
        spec=pinecone.ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(index_name)

import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}

def get_repo_info(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_repo_files(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_file_content(owner, repo, path):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(url, headers=HEADERS)
    if response.ok:
        file_content = response.json().get('content', '')
        file_content = base64.b64decode(file_content).decode('utf-8')  # Decode from base64 and then to string
        return file_content
    return None

import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI API with ChatOpenAI
openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# Initialize Embeddings
embeddings = OpenAIEmbeddings()

# Connect Pinecone to LangChain
vector_store = Pinecone(
    embedding=embeddings,  # Pass the Embeddings object directly
    index=index,           # Ensure this is the Pinecone index instance
    text_key="text"        # Specify the key for text in the data dictionary
)

def summarize_repository(repo_data, readme_content):
    # Truncate readme_content to fit within token limits
    max_token_length = 3000
    if len(readme_content) > max_token_length:
        readme_content = readme_content[:max_token_length]

    messages = [
        SystemMessage(content="You are a highly knowledgeable assistant."),
        HumanMessage(content=f"""
        Summarize the following GitHub repository, focusing on its main purpose, key features, and the overall goals of the project. Do not include a 'Getting Started for New Contributors' section.

        Repository Details:
        - Name: {repo_data['name']}
        - Description: {repo_data['description']}
        - README Content: {readme_content}
        """)
    ]
    response = openai_llm(messages)
    return response.content


def answer_question(query, context):
    # Truncate context to fit within token limits
    max_token_length = 3000
    if len(context) > max_token_length:
        context = context[:max_token_length]

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"You are assisting a developer working with the following GitHub repository. Answer their question based on the repository content provided.\n\nRepository Context: {context}\nQuestion: {query}")
    ]
    response = openai_llm(messages)
    return response.content

def generate_suggestions(code_snippet):
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=f"Analyze the following code from a GitHub repository and suggest improvements. Consider code quality, efficiency, readability, and best practices.\n\nCode Snippet:\n{code_snippet}")
    ]
    response = openai_llm(messages)
    return response.content

def get_repo_files(owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}/contents"
    response = requests.get(url, headers=HEADERS)
    return response.json()

def get_all_file_contents(owner, repo):
    files = get_repo_files(owner, repo)
    file_contents = {}
    
    if isinstance(files, list):
        for file in files:
            # Check if the file is a code file
            if file['type'] == 'file' and file['name'].split('.')[-1] in ['py', 'js', 'html', 'css', 'java', 'cpp', 'md']:  # Add extensions as needed
                content = get_file_content(owner, repo, file['path'])
                if content:
                    file_contents[file['name']] = content
    return file_contents

def verify_answer(query, initial_answer, context):
    # Prompt to ask the model to review and correct its answer
    verification_prompt = f"""
    Here is a question and the answer that was generated based on the context of a GitHub repository. Please review the answer for accuracy and correctness. If there are any mistakes, provide a corrected version. If the answer is accurate, confirm that it is correct.

    Context:
    {context}

    Question: {query}
    Initial Answer: {initial_answer}

    Please provide your review and corrections if needed:
    """
    
    messages = [
        SystemMessage(content="You are a highly knowledgeable assistant."),
        HumanMessage(content=verification_prompt)
    ]
    response = openai_llm(messages)
    return response.content

def answer_question(query, context):
    # Generate the initial answer
    max_token_length = 3000
    if len(context) > max_token_length:
        context = context[:max_token_length]

    initial_prompt = f"""
    Based on the following context, please answer the question accurately:

    Context:
    {context}

    Question: {query}

    Answer:
    """
    
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=initial_prompt)
    ]
    initial_answer = openai_llm(messages).content

    # Optionally verify the initial answer, but we will return the original answer
    # corrected_answer = verify_answer(query, initial_answer, context)
    
    # Return the actual initial answer without any review or verification
    return initial_answer


# Example of using the self-correcting function in the main loop
def main():
    owner = "feder-cr"
    repo = "linkedIn_auto_jobs_applier_with_AI"
    
    # Fetch repository information and README content
    repo_data = get_repo_info(owner, repo)
    file_contents = get_all_file_contents(owner, repo)
    
    all_texts = []
    for filename, content in file_contents.items():
        print(f"Processing file: {filename}")
        
        # Summarize if it's a markdown file, otherwise just store the content
        if filename.endswith(".md"):
            summary = summarize_repository(repo_data, content)
            print(f"Summary for {filename}:")
            print(summary)
            all_texts.append(summary)
        else:
            all_texts.append(content)
    
    # Store all file contents and summaries in Pinecone
    vector_store.add_texts(all_texts)
    
    # Interactive Q&A Loop
    print("\nYou can now ask questions about the repository. Type 'exit' to quit.")
    while True:
        question = input("\nAsk your question: ")
        if question.lower() == 'exit':
            print("Exiting Q&A. Goodbye!")
            break
        
        # Here we can query all file contents
        context = "\n".join(all_texts)
        answer = answer_question(question, context)
        print(f"Answer: {answer}")

if __name__ == "__main__":
    main()


