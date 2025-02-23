from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq # LLM model
from langchain_community.vectorstores import FAISS # Vector Embeddings
from langchain_community.document_loaders import WebBaseLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import bs4
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
import csv
import requests
import os
from bs4 import BeautifulSoup
import faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from uuid import uuid4
import httpx
import urllib3
urllib3.disable_warnings()
from dotenv import load_dotenv
load_dotenv()

#os.environ["GROQ_API_KEY"] = os.getenv("Groq_api_key")
os.environ["HuggingFace_API_KEY"] = os.getenv("HuggingFace_API_KEY")

system_template = """

        Answer the user's questions based on the below context.
        If the context doesn't contain any relevant information to the question, don't make something up and just only respond back with "I don't know" :

        <context>
        {context}
        </context>
        
        """
class Rag():
    
    def __init__(self, url):
        self.url = url
        self.system_template = system_template
        self.docs = []
        self.vector_store = None
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")  # Load embeddings once

    def web_parsing(self):
        print(self.url)
        page = requests.get(self.url)
        soup = BeautifulSoup(page.content, 'html.parser')
        text = soup.get_text()
        metadata = {"source": self.url}
        if title := soup.find("title"):
            metadata["title"] = title.get_text()
        if description := soup.find("meta", attrs={"name": "description"}):
            metadata["description"] = description.get("content", "No description found.")
        if html := soup.find("html"):
            metadata["language"] = html.get("lang", "No language found.")

        self.docs.append(Document(page_content=text, metadata=metadata))
        return self.docs

    def create_embeddings(self, document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        all_splits = text_splitter.split_documents(document)
        index = faiss.IndexFlatL2(len(self.embeddings.embed_query("hello world")))
        self.vector_store = FAISS(embedding_function=self.embeddings, index=index,
                                  docstore=InMemoryDocstore(),
                                  index_to_docstore_id={})
        uuids = [str(uuid4()) for _ in all_splits]

        _acrice = self.vector_store.add_documents(all_splits, document_ids=uuids)
        if _acrice:
            
            self.vector_store.save_local("Faiss_index_new")
            
        return self.vector_store

    def response(self, messages: dict):
        documents = self.web_parsing()
        if os.path.exists("Faiss_index_new"):
            self.vector_store = FAISS.load_local("Faiss_index_new", embeddings=self.embeddings, allow_dangerous_deserialization=True)
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        else:
            vector_store = self.create_embeddings(documents)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGroq(model='llama-3.1-8b-instant', api_key=os.getenv("Groq_api_key"))
        
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_template,
                ),
                MessagesPlaceholder(variable_name="message"),
            ]
        )
        document_chain = create_stuff_documents_chain(llm, question_answering_prompt)
        query_transform_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="message"),
                (
                    "user",
                    "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
                ),
            ]
        )

        query_transforming_retriever_chain = RunnableBranch(
            (
                lambda x: len(x.get("message", [])) == 1,
                # If only one message, then we just pass that message's content to retriever
                (lambda x: x["message"][-1]['content']) | retriever,
            ),
            # If messages, then we pass inputs to LLM chain to transform the query, then pass to retriever
            query_transform_prompt | llm | StrOutputParser() | retriever,
            ).with_config(run_name="chat_retriever_chain")

        conversational_retrieval_chain = RunnablePassthrough.assign(
            context=query_transforming_retriever_chain,
            ).assign(
            answer=document_chain,
            )
        response = conversational_retrieval_chain.invoke(messages)
        return response


