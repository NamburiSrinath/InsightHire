# Read the content of the Uday_Girish_Maradana_Resume2P.pdf file
resume_reader = SimpleDirectoryReader(input_files=["./Uday_Girish_Maradana_Resume2P.pdf"])
resume_document = resume_reader.load_data()

# print(resume_document[0].metadata['file_name'])


storage_name = "./storage/" + resume_document[0].metadata['file_name']

try:
    storage_context = StorageContext.from_defaults(
        persist_dir=storage_name
    )
    resume_index = load_index_from_storage(storage_context)

    index_loaded = True
except:
    index_loaded = False
    

if not index_loaded:
    # load data
    resume_doc = SimpleDirectoryReader(
        input_files=["./Uday_Girish_Maradana_Resume2P.pdf"]
    ).load_data()

#build index
resume_index = VectorStoreIndex.from_documents(resume_doc, show_progress=True)

#persist index
resume_index.storage_context.persist(persist_dir=storage_name)
index_loaded = True

resume_eval_engine = resume_index.as_query_engine(similarity_top_k=50, llm=llm)

query_engine_tools = [
    QueryEngineTool(
        query_engine=resume_eval_engine,
        metadata=ToolMetadata(
            name="resume_jd_analyzer",
            description=(
                "Seasoned evaluator of resumes against job descriptions. It has context of both the resume and the job description. You need to carefully analyze the resume against the job description and evaluate how good the candidate by understanding their experience, skills, projects and evaluate against the json output of the job description analyzer. "
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    ),
]

agent = ReActAgent.from_tools(
    query_engine_tools,
    system_prompt=r"You are a seasoned evaluator of resumes against job descriptions. You have context of both the resume and the job description. You need to carefully analyze the resume against the job description and evaluate how good the candidate by understanding everything on their resume - experience, skills, projects. Break down each metric in the requirements into individual metrics and for each metric score the user on the scale of 10 along with giving. An insight for why that score based on the resume. return the response as a json. Give grounded insights with skill, score, and insight. Example output: {\"Python\": {\"score\": 8, \"insight\": \"The candidate has experience in Python and Java, which are relevant to the job description.\"}, \"Pytorch\": {\"score\": 7, \"insight\": \"The candidate has experience PyTorch which are relevant to the job description.\"}, \"GenAI\": {\"score\": 8, \"insight\": \"The candidate has experience using LLMs, OpenAI, demonstrating strong skills in GenAI.\"}}",                                                                               
    llm=llm,
    verbose=True,
    max_turns=10,
)



# Query the agent with the user input
response1 = agent.query("List the candidate evaluation using json. Give the example output: {\"Python\": {\"score\": 8, \"insight\": \"The candidate has experience in Python and Java, which are relevant to the job description.\"}, \"Pytorch\": {\"score\": 7, \"insight\": \"The candidate has experience PyTorch which are relevant to the job description.\"}, \"GenAI\": {\"score\": 8, \"insight\": \"The candidate has experience using LLMs, OpenAI, demonstrating strong skills in GenAI.\"}}.")

# Print the response
print(response1)