# InsightHire

This was done as part of Hackathon. 

### Key Idea
Our product takes input an external Job Description, Resume.

This can be mainly useful for recruiters, hiring managers, HRs or panel people to understand the strengths and weaknessess of a candidate so the interview questions can be "candidate specific"

We mainly have 4 agents
- Job Description Agent (jd_agent.py) - Given the external and internal job description, this will generate the key skills needed to look for a candidate
- Github Agent (github_agent.py) - Given a Github profile and the LLM summarized job description (from `jd_agent.py`), we try to scrap the contents from it and generate human-friendly evaluation response on the required criteria. 
- Google Scholar Agent (`scholar_agent.py`) - Given a Google Scholar profile and the LLM summarized job description (from `jd_agent.py`) we try to scrap the contents from it and generate human-friendly evaluation response on the required criteria.
- Resume agent (`resume_agent.py`) - Given a resume and the LLM summarized job description (from `jd_agent.py`), evaluate how good the particular candidate is.
- Aggregator Agent (`aggregator_agent.py`) - This combines the information from all the previous agents to provide a final evaluation based on a predefined metric.

### Steps

1. Look at `presentation.pptx` to understand the whole motivation and the entire system design.
2. To run the code, `python main.py`

You need OpenAI Key (place in `.env` with `OPENAI_KEY`) to run this code.

As of now, we observed that the performance of this agent is not satisfactory and there're lots of scope of improvement.