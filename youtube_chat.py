from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Cohere
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

COHERE_API_KEY = "_____"

text_embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)


def create_db_from_video_url(video_url):
    doc_loader = YoutubeLoader.from_youtube_url(video_url)
    vid_transcript = doc_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(vid_transcript)

    vector_db = FAISS.from_documents(docs, text_embeddings)
    return vector_db

def get_query_response(vector_db, query, num_docs=4):
    docs = vector_db.similarity_search(query, k=num_docs)
    docs_page_content = " ".join([d.page_content for d in docs])

    model = Cohere(cohere_api_key=COHERE_API_KEY, temperature=0, truncate="END", stop=["Human"])

    prompt_template = """
        You are a helpful and informative assistant that that can answer questions about youtube videos 
        based on the video's transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed while being able to be understood by a general audience.
        """
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(prompt_template)

    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=model, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")
    return response, docs

def main():
    vid_url = input("Enter the URL of your desired YouTube video: ")
    query = ""

    db = create_db_from_video_url(vid_url)

    while True:
        print("Welcome to the YouTube Bot!")
        query = input("Enter your query, or \"exit\". ")

        if query.lower() == "exit":
            break
        else:
            response = get_query_response(db, query)[0]
            print(response)


if __name__ == "__main__":
    main()
