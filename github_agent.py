import json
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# Load GitHub profile JSON file
def load_github_profile(file_path):
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
def generate_evaluation_prompt(user_data, requirements):
    prompt = f"""
    Evaluate the following GitHub profile based on the skill requirements.

    **GitHub Profile:**
    - Username: {user_data["username"]}
    - Name: {user_data["name"]}
    - Followers: {user_data["followers"]}
    - Following: {user_data["following"]}
    - Public Repositories: {user_data["public_repos_count"]}
    - Total Commits: {user_data["total_commits"]}
    
    **Repositories:**
    {json.dumps(user_data["repo_details"], indent=4)}

    **Requirements:**
    {requirements}
    Break down each metric in the requirements into individual metrics and for each metric score the user on the scale of 10 along with giving.
    An insight for why that score based on the github contributions. return the response as a json.
    give grounded insights with repository names, number of commits, types of contributions etc for each insight. Be sure to mention extensive details.

    Example output:
    "Python": {{"score": 5, "insight":"the user has strong contributions to github repositories where python was used"}},
    "C++": {{"score": 2.5, "insight": "the user has no commits to repositories with C++. However, the user has many repositories with C++."}},
    "GenAI": {{"score": 0, "insight": "there are no repositories that have any relation to generative AI"}}

    """

    return prompt



# Call OpenAI ChatGPT API
def get_chatgpt_evaluation(prompt, user_data):


    general_user_info=f"""
    **GitHub Profile:**
    - Username: {user_data["username"]}
    - Name: {user_data["name"]}
    - Followers: {user_data["followers"]}
    - Following: {user_data["following"]}
    - Public Repositories: {user_data["public_repos_count"]}
    - Total Commits: {user_data["total_commits"]}
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are an expert recruiter analyzing GitHub profiles."},
                  {"role": "user", "content": prompt}],
        temperature=0.2,
    )

    return general_user_info + response.choices[0].message.content 


def run_github_agent(requirements=None):
    file_path = "github_docs/udaygirish.json"  # Update with actual JSON file path
    
    # openai.api_key = api_key
    if requirements==None:
        requirements="""    - Languages: Python, C++
            - Frameworks: PyTorch
            - Fields of Expertise: GenAI, LLMs, Computer Vision, Multi-GPU Training
            - Culture: Strong individual contributor, works well in cross-collaboration.
            """
    # Load and process the GitHub profile
    profile_data = load_github_profile(file_path)
    user_data = extract_user_data(profile_data)

    # Generate prompt and evaluate
    prompt = generate_evaluation_prompt(user_data, requirements)
    evaluation = get_chatgpt_evaluation(prompt, user_data)
    return evaluation

# Main function
def main():
    evaluation = run_github_agent(requirements=requirements)
    # print("\n===== Candidate Evaluation based on Github Agent =====\n")
    # print(evaluation)
    return evaluation

if __name__ == "__main__":
    main()
