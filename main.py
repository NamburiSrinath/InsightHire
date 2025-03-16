from github_agent import run_github_agent
from scholar_agent import run_scholar_agent
from jd_agent import run_jd_requirements
from resume_agent import run_resume_agent
from aggregator_agent import run_aggregator_agent

def main():
    jd_requirements = run_jd_requirements()
    print("JD Requirements")
    print(jd_requirements)
    print("="*50)

    github_data = run_github_agent(requirements=jd_requirements)
    print("Github evaluation")
    print(github_data)
    print("="*50)
    
    scholar_data = run_scholar_agent(requirements=jd_requirements)
    print("Google Scholar evaluation")
    print(scholar_data)
    print("="*50)

    resume_data = run_resume_agent()
    print("Resume evaluation")
    print(resume_data)
    print("="*50)

    final_evaluation = run_aggregator_agent(github_data, scholar_data, resume_data, jd_requirements)
    print("============ Candidates final evaluation ============")
    print(final_evaluation)
if __name__ == "__main__":
    main()