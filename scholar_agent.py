import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

from llama_index.core import (
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

from os import environ
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = environ["OPENAI_API_KEY"]

class ScholarAgent:
    def __init__(self, requirements, papers_dir: str, persist_dir: str = "./storage/scholar_index"):
        self.papers_dir = papers_dir
        self.persist_dir = persist_dir
        self.documents = None
        self.index = None
        self.query_engine = None
        self.agent = None
        self.document_metadata_cache = {}  # Cache for document metadata
        # Parse requirements once during initialization
        # self.requirements = """Requirements:
        #                 Languages: Python, C++
        #                 Frameworks: PyTorch
        #                 Fields of Expertise: GenAI, LLMs, Computer Vision, Multi-GPU Training
        #                 Culture: Strong individual contributor, works well in cross-collaboration."""
        self.requirements = requirements
        self.parsed_requirements = self._parse_requirements(self.requirements)
        self.all_skills = self._extract_all_skills(self.parsed_requirements)
        
    def _parse_requirements(self, requirements_text):
        """Parse the requirements text into a structured format."""
        lines = [line.strip() for line in requirements_text.split('\n') if line.strip()]
        parsed_req = {}
        
        for line in lines:
            if ":" in line:
                category, skills = line.split(":", 1)
                category = category.strip()
                skills_list = [skill.strip() for skill in skills.split(',')]
                parsed_req[category] = skills_list
        
        return parsed_req
    
    def _extract_all_skills(self, parsed_requirements):
        """Extract all individual skills as a flat list."""
        all_skills = []
        for category, skills in parsed_requirements.items():
            all_skills.extend(skills)
        return all_skills
        
    def load_documents(self):
        """Load all documents from the papers directory."""
        # print("Loading documents...")
        self.documents = SimpleDirectoryReader(self.papers_dir).load_data()
        return self.documents
        
    def load_or_build_index(self):
        """Load index from disk if available, otherwise build and save it."""
        if os.path.exists(self.persist_dir):
            # print("Loading existing index...")
            storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
            self.index = load_index_from_storage(storage_context)
        else:
            # print("Building new index...")
            if self.documents is None:
                self.load_documents()
                
            # Split documents into manageable chunks
            node_parser = SentenceSplitter(chunk_size=1024)
            nodes = node_parser.get_nodes_from_documents(self.documents)
            
            # Create index
            self.index = VectorStoreIndex(nodes)
            
            # Save index to disk
            self.index.storage_context.persist(persist_dir=self.persist_dir)
        
        return self.index
    
    def extract_paper_metadata(self, author_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Extract metadata from all papers, optionally filtering by author.
        Returns a list of dictionaries containing title and key technologies for each paper.
        """
        if self.documents is None:
            self.load_documents()
            
        # Create an LLM for metadata extraction
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.4)
        
        all_papers_metadata = []

        # Process each document once
        for doc in self.documents:
            # Extract the filename as a reference
            file_path = doc.metadata.get("file_path", "unknown")
            filename = os.path.basename(file_path)
            
            # Check if we already processed this document
            if file_path in self.document_metadata_cache:
                cached_metadata = self.document_metadata_cache[file_path]
                
                # If author filter is specified, check if this document belongs to that author
                if author_name and not cached_metadata.get("has_author", {}).get(author_name, False):
                    continue
                    
                all_papers_metadata.append(cached_metadata["metadata"])
                continue
                
            # print(f"Analyzing {filename}")
            
            # Initialize cache entry for this document
            if file_path not in self.document_metadata_cache:
                self.document_metadata_cache[file_path] = {
                    "metadata": {},
                    "has_author": {}
                }
            
            # Check if this document mentions the author (if specified)
            author_present = True
            if author_name:
                # Check if we already determined author presence
                if author_name in self.document_metadata_cache[file_path]["has_author"]:
                    author_present = self.document_metadata_cache[file_path]["has_author"][author_name]
                else:
                    # print(f"Checking if {author_name} is mentioned in {filename}")
                    author_check_prompt = f"""
                    Does the academic paper in the following content mention {author_name} as an author? 
                    Answer only 'yes' or 'no'.
                    
                    Content:
                    {doc.text[:1000]}  # Check beginning of document where authors are usually mentioned
                    """
                    author_result = llm.complete(author_check_prompt)
                    author_present = "yes" in author_result.text.lower()
                    # Cache the result
                    self.document_metadata_cache[file_path]["has_author"][author_name] = author_present
            
            if not author_present:
                continue
                
            # Extract metadata if we haven't done it yet
            if not self.document_metadata_cache[file_path]["metadata"]:
                # Extract title
                title_prompt = """
                Extract the title of the academic paper from the following content.
                Return only the title, nothing else.
                
                Content:
                """ + doc.text[:2000]  # Title is usually at the beginning
                
                title_result = llm.complete(title_prompt)
                title = title_result.text.strip().strip('"')
                
                # Extract key technologies/techniques
                tech_prompt = """
                Identify the key technologies, methods, algorithms, or techniques described in this academic paper.
                List up to 5 key technologies or approaches as a comma-separated list.
                
                Content:
                """ + doc.text
                
                tech_result = llm.complete(tech_prompt)
                key_technologies = [tech.strip() for tech in tech_result.text.split(',')]
                
                # Cache the metadata
                self.document_metadata_cache[file_path]["metadata"] = {
                    "file": filename,
                    "title": title,
                    "key_technologies": key_technologies
                }
            
            # Add to results
            all_papers_metadata.append(self.document_metadata_cache[file_path]["metadata"])
        
        # print(f"Found {len(all_papers_metadata)} papers" + (f" by {author_name}" if author_name else ""))
        return all_papers_metadata
    
    def format_response_as_json(self, analysis_text: str) -> str:
        """Format the agent's analysis response as a structured JSON with the exact required skill keys."""
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.4)
        
        skills_list = ", ".join([f'"{skill}"' for skill in self.all_skills])
        
        format_prompt = f"""
        Convert the following skill analysis into a properly formatted JSON object.
        
        The JSON MUST include each of these exact skills as keys: {skills_list}
        
        Each skill should have an object value containing "score" (integer 0-10) and "insight" (string).
        The insight should be specific and mention relevant publications and technologies.
        
        Analysis text:
        {analysis_text}
        
        Return ONLY valid JSON with no explanations or markdown formatting. Example format:
        {{
          "Python": {{"score": 8, "insight": "Author has multiple Python projects including..."}},
          "C++": {{"score": 6, "insight": "Author has contributed to C++ projects with papers on..."}}
        }}
        
        IMPORTANT: Make sure EVERY skill from the provided list is included in the JSON, even if with a score of 0.
        """
        
        result = llm.complete(format_prompt)
        # Validate JSON and fix if needed
        try:
            json_obj = json.loads(result.text)
            # Ensure all required skills are present
            for skill in self.all_skills:
                if skill not in json_obj:
                    json_obj[skill] = {"score": 0, "insight": "No evidence found in the publications."}
            return json.dumps(json_obj, indent=2)
        except json.JSONDecodeError:
            # If not valid JSON, try to extract just the JSON part
            try:
                # Look for text between curly braces
                json_match = re.search(r'\{.*\}', result.text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    json_obj = json.loads(json_str)
                    # Ensure all required skills are present
                    for skill in self.all_skills:
                        if skill not in json_obj:
                            json_obj[skill] = {"score": 0, "insight": "No evidence found in the publications."}
                    return json.dumps(json_obj, indent=2)
                else:
                    return '{"error": "Could not parse response as JSON"}'
            except:
                return '{"error": "Could not parse response as JSON"}'
    
    def build_agent(self):
        """Build ReActAgent with paper analysis tools."""
        # Ensure documents are loaded
        if self.documents is None:
            self.load_documents()
            
        # Load or build index for general querying
        self.load_or_build_index()
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            response_mode="compact",
            similarity_top_k=3
        )
        
        # Create general query tool
        query_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="paper_search",
                description="Useful for searching information across all papers."
            )
        )
        
        # Create paper metadata extraction tool
        def get_papers_by_author(author_name: str) -> Dict[str, Any]:
            """Extract metadata from all papers by a specific author."""
            papers = self.extract_paper_metadata(author_name)
            
            return {
                "author": author_name,
                "paper_count": len(papers),
                "papers": papers
            }
        
        author_papers_tool = FunctionTool.from_defaults(
            fn=get_papers_by_author,
            name="author_papers_extractor",
            description="Extracts titles and key technologies from all papers authored by a specific person."
        )
        
        # Create a function to format the final response as JSON
        format_json_tool = FunctionTool.from_defaults(
            fn=self.format_response_as_json,
            name="format_json",
            description="Formats the final analysis as a properly structured JSON object with the exact required skill keys."
        )
        
        # Create ReActAgent
        llm = OpenAI(model="gpt-3.5-turbo", temperature=0.4)

        self.agent = ReActAgent.from_tools(
            [query_tool, author_papers_tool, format_json_tool],
            llm=llm,
            verbose=False,
            system_prompt=f"""
            You are an expert at evaluating research profiles based on their publications.
            
            Evaluate the following profile based on the skill requirements.

            **Requirements:**
            {self.requirements}
            
            For each specific skill mentioned in the requirements, evaluate the candidate and assign a score from 0-10,
            where 0 means no evidence of skill and 10 means exceptional proficiency.
            
            The exact skills to evaluate are:
            {", ".join(self.all_skills)}
            
            For each skill, provide detailed insights explaining the score, mentioning specific publications,
            technologies, and relevant evidence from the papers.
            
            IMPORTANT: After completing your analysis, you MUST call the format_json tool to convert your evaluation 
            into a proper JSON format where:
            1. Each key is one of the specific skills listed above
            2. Each value is an object with "score" (integer 0-10) and "insight" (detailed string)
            
            Be specific and provide detailed insights with publication titles and keywords from the publications.
            """
        )
        
        return self.agent
    
    def query(self, question: str) -> str:
        """Query the agent with a question."""
        if self.agent is None:
            self.build_agent()
            
        response = self.agent.query(question)
        
        # Check if response is already JSON
        try:
            json_obj = json.loads(str(response))
            # Ensure all required skills are present in the JSON
            for skill in self.all_skills:
                if skill not in json_obj:
                    json_obj[skill] = {"score": 0, "insight": "No evidence found in the publications."}
            return json.dumps(json_obj, indent=2)  # Return formatted JSON
        except json.JSONDecodeError:
            # If not JSON, try to format it using our formatter
            return self.format_response_as_json(str(response))

def run_scholar_agent(requirements):
    scholar_agent = ScholarAgent(requirements, "/hdd4/srinath2/seattle_hackathon/scholar_docs")
    # Example query about a specific author's papers
    question = "Analyze the papers authored by the candidate?"
    # print(f"\nQuestion: {question}")
    scholar_answer = scholar_agent.query(question)
    return scholar_answer

def main():
    scholar_answer = run_scholar_agent(requirements=requirements)
    # print("\n===== Candidate Evaluation based on Scholar Agent =====\n")
    # print(f"Answer: {scholar_answer}")
    return scholar_answer

# Example usage
if __name__ == "__main__":
    main()