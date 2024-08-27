from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from pinecone import Pinecone

from dotenv import load_dotenv
load_dotenv()
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)
host = "https://freechat-production-hbcu99b.svc.aped-4627-b74a.pinecone.io"
pinecone_index = pc.Index(host=host)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.environ.get("OPENAI_API_KEY"))  # "text-embedding-3-large"
# Create Pinecone index and vector store
namespace = ".."  # 12-step materials
vector_store = PineconeVectorStore(
    index=pinecone_index,
    namespace=namespace,
    embedding=embeddings,
    text_key="window"
)
retriever = vector_store.as_retriever()
namespace = "prompts"  # Expert prompts
prompt_vector_store = PineconeVectorStore(
    index=pinecone_index,
    namespace=namespace,
    embedding=embeddings,
    text_key="Response"
)
prompt_retriever = prompt_vector_store.as_retriever()
namespace = "recommendation"  # Recommendations
recommendations_vector_store = PineconeVectorStore(
    index=pinecone_index,
    namespace=namespace,
    embedding=embeddings,
    text_key="recommendations"
)
recommendations_retriever = recommendations_vector_store.as_retriever()
namespace = "recommendation_quotes"  # Quote recommendations
quote_vector_store = PineconeVectorStore(
    index=pinecone_index,
    namespace=namespace,
    embedding=embeddings,
    text_key="_node_content"
)
quote_retriever = quote_vector_store.as_retriever()


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_12_steps_excerpts",
    "Use this tool to retrieve excerpts from 12 Steps literatures to answer specific questions. Use a detailed plain text question as input to the tool."
)

prompt_retriever_tool = create_retriever_tool(
    prompt_retriever,
    "retrieve_prompts",
    "Search and return few-shot examples prompts based on user's query and user info.",
)

recommendations_retriever_tool = create_retriever_tool(
    recommendations_retriever,
    "retrieve_recommendations",
    "Search and return Content Recommendations based on user's current step and program.",
)

quote_retriever_tool = create_retriever_tool(
    quote_retriever,
    "retrieve_quotes",
    "Search and return quotes based on user's query whenever user needs inspiration or motivation.",
)

# print(retriever_tool.invoke(
#     "What is the first step of the 12-step program?"))

tools = [
    retriever_tool,
    prompt_retriever_tool,
    recommendations_retriever_tool,
    # quote_retriever_tool,
    # TavilySearchResults(max_results=1)
]

#tools = [TavilySearchResults(max_results=1)]
