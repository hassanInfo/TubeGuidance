from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url):
    """
    Create a document database from the transcript of a YouTube video.

    Parameters:
    - video_url (str): The URL of the YouTube video.

    Returns:
    - db: A document database constructed from the video transcript.
    """
    # Load transcript from the YouTube video
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # Split the transcript into documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # Create a document database with FAISS indexing
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(db, query, k=4):
    """
    Retrieve a response to a query using a document database and language model.

    Parameters:
    - db: The document database.
    - query (str): The user's query.
    - k (int): The number of documents to consider in the response. Default is 4.

    Returns:
    - response (str): The generated response to the query.
    - docs: The documents considered in generating the response.
    """
    # Perform similarity search in the document database
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize OpenAI language model
    llm = OpenAI(model_name="text-davinci-003", temperature=0.8)

    # Define a prompt template for generating responses
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
            You're an informative assistant specializing in responding to queries about YouTube videos. 
            Your expertise is grounded in the transcripts of these videos.

            Address the following inquiry: {question}
            Delve into the video transcript for insights: {docs}

            Rely solely on factual content within the transcript to construct your response.

            If you find yourself lacking adequate information for a response, simply state "I lack the information needed to provide an answer."

            Your replies should be thorough and detailed.
            """)

    # Initialize an LLMChain for response generation
    chain = LLMChain(llm=llm, prompt=prompt)

    # Generate response using the language model chain
    response = chain.run(question=query, docs=docs_page_content)
    
    # Clean up response
    response = response.replace("\n", "")

    return response, docs
