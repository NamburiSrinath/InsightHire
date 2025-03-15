import json
from openai import OpenAI
import os

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Load GitHub profile JSON file
def load_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Extract relevant insights from the GitHub profile
def extract_user_data(profile):
    user_data = {
        "username": profile.get("user"),
        "name": profile.get("name"),
        "followers": profile.get("followers"),
        "following": profile.get("following"),
        "public_repos_count": profile.get("public_repos_count"),
        "total_commits": profile.get("total_commits_in_public_repos_by_user"),
        "repo_details": [],
    }

    for repo in profile.get("repo_details", []):
        repo_data = {
            "repo_name": repo.get("repo_name"),
            "languages": list(repo.get("languages", {}).keys()),
            "commits": [commit["message"] for commit in repo.get("commits", [])],
        }
        user_data["repo_details"].append(repo_data)

    return user_data

# Generate a prompt for evaluating the user's experience
def generate_evaluation_prompt(github_data, scholar_data, resume_data, requirements):
    prompt = f"""
    Given the following evaluation from a github agent: {github_data}

    and the evaluation of paper publications agent: {scholar_data}

    and evaluation of resume agent: {resume_data}

    **Requirements:**
    {requirements}

    Generate a final score for the candidate based on the inputs from github and paper publications from the requirements.

    Based on the results, generate a set of questions evaluating these areas from the paper publications or github or resume for the candidate to evaluate his areas of strengths and weaknesses.
    return output in json format.

    Example output:
    "Python": {{"score": 5, "insight":"the user has strong contributions to github repositories where python was used"}},
    "C++": {{"score": 2.5, "insight": "the user has no commits to repositories with C++. However, the user has many repositories with C++."}},
    "GenAI": {{"score": 0, "insight": "there are no repositories that have any relation to generative AI"}}
    "weakness_evaluation": {{"GenAI": ["what role does context length play in user chat experience for chatbots",...]}}
    "strength_evaluation": {{...}}
    """

    return prompt



# Call OpenAI ChatGPT API
def get_chatgpt_evaluation(prompt):


    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an expert recruiter analyzing GitHub profiles."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content 


def run_aggregator_agent(github_data=None, scholar_data=None, resume_data=None, requirements=None):
    # file_path = "udaygirish.json"  # Update with actual JSON file path
    
    # openai.api_key = api_key
    if requirements==None:
        requirements="""    - Languages: Python, C++
            - Frameworks: PyTorch
            - Fields of Expertise: GenAI, LLMs, Computer Vision, Multi-GPU Training
            - Culture: Strong individual contributor, works well in cross-collaboration.
            """
    # Load and process the GitHub profile
    # profile_data = load_github_profile(file_path)
    # user_data = extract_user_data(profile_data)
    if github_data==None:
        with open('github_data.txt', 'r') as file:
            github_data = file.read()
    if scholar_data==None:
        with open('scholar_data.txt', 'r') as file:
            scholar_data = file.read()
    if resume_data==None:
        with open('resume_data.txt', 'r') as file:
            resume_data = file.read()


    # Generate prompt and evaluate
    prompt = generate_evaluation_prompt(github_data, scholar_data,resume_data, requirements)
    evaluation = get_chatgpt_evaluation(prompt)
    return evaluation

# Main function
def main():
    evaluation = run_aggregator_agent(github_data, scholar_data, resume_data, requirements)
    print("\n===== Candidate Evaluation =====\n")
    print(evaluation)

if __name__ == "__main__":
    main()
