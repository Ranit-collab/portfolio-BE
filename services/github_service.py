import requests

def fetch_github_data(username: str):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code != 200:
        return "Failed to fetch GitHub data"
    
    repos = response.json()
    summary = []
    for repo in repos:
        summary.append(f"{repo['name']}: {repo.get('description', 'No description')} (⭐ {repo['stargazers_count']})")
    return "\n".join(summary)
