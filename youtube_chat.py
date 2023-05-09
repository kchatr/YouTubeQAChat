from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import Cohere
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import textwrap

text_embeddings = CohereEmbeddings()

video_url = "https://youtu.be/k7RM-ot2NWY"

def create_db_from_video_url(video_url):
    doc_loader = YoutubeLoader.from_youtube_url(video_url)
    vid_transcript = doc_loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(vid_transcript)

    vector_db = FAISS.from_documents(docs, text_embeddings)
    return vector_db