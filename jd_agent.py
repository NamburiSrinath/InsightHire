from dotenv import load_dotenv
import os

# Set the OpenAI API key
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI


# Create an llm object to use ]for the QueryEngine and the ReActAgent
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.01)

try:
    storage_context = StorageContext.from_defaults(
        persist_dir="./storage/agg_jd"
    )
    jd_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False
    
# load data
jd_docs = SimpleDirectoryReader(
    input_files=["JD_Docs/InternalAppliedScientistJD.docx", "JD_Docs/ExternalAppliedScientistJD.docx"]
).load_data()

#build index
jd_index = VectorStoreIndex.from_documents(jd_docs, show_progress=True)

#persist index
jd_index.storage_context.persist(persist_dir="./storage/agg_jd")
index_loaded = True

jd_engine = jd_index.as_query_engine(similarity_top_k=5, llm=llm)

query_engine_tools = [
    QueryEngineTool(
        query_engine=jd_engine,
        metadata=ToolMetadata(
            name="jd_analyzer",
            description=(
                "Seasoned job description analyzer. It has context of both both internal and external job descriptions. You need to carefully analyze the job description and give information about the required skills (such as coding languages, frameworks), field of expertise (such as facial recognition, model compression), and culture (if they are individual contributors, travel etc.). "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = ReActAgent.from_tools(
    query_engine_tools,
    system_prompt=r"You are a seasoned job description analyzer. You have context of both the internal and external job descriptions. The internal job descriptions are used by the interviewer and the hiring panel to better align with the skills for the role. The external job description is used by external interviewees and candidates to understand what the role is looking for. You need to carefully analyze job descriptions and give information about the required technical and non-technical skills for the job. The output should be in a json. ",
    llm=llm,
    verbose=False,
    max_turns=10,
)

def run_jd_requirements():
    # Query the agent with the user input

    response = agent.query("List all the required technical and non-technical skills using json. the key areas we are looking for are: [\"Programming languages\", \"Frameworks\",\"Field of expertise\",\"personal skills\"] For example: {\"Languages\": [\"python\", \"java\"], \"Frameworks\": [\"Langchain\", \"PyTorch\"], \"field_of_expertise\": [\"facial recognition\", \"model compression\"], \"personal skills\": [\"individual contributor\", \"cross collaboration with engineering teams\, \"cross collaboration with product teams\"]}.")

    return response

if __name__ == "__main__":
    run_jd_requirements()