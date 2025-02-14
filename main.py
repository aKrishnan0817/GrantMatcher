from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from grant_analysis import analyze_grants, evaluate_project  # Import from your file
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables to store pre-computed data
clusters = {}
cluster_labels = {}
labels = []
grant_descriptions = [] 

@app.on_event("startup")
async def startup_event():
    global clusters, cluster_labels, labels, grant_descriptions  # Access global variables

    #  grant descriptions (REPLACE WITH  ACTUAL DATA)
    grant_descriptions = [
        "This grant supports innovative research in renewable energy with a focus on community-driven sustainability projects.",
        "Funding opportunity for digital education innovation and the development of cutting-edge online learning platforms.",
        "This grant is dedicated to advancing healthcare research, focusing on improving patient outcomes and innovative medical technologies.",
        "Supports community development projects that promote local economic growth and social entrepreneurship.",
        "Grants for research in artificial intelligence, machine learning, and their applications in real-world problem solving.",
        # ... more grant descriptions
    ]

    clusters, cluster_labels, labels = analyze_grants(grant_descriptions)  # Run analysis on startup
    print("Grant analysis completed during startup.") #Confirmation message


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Grant Proposal Evaluation</title>
        <link rel="stylesheet" href="/static/style.css">
    </head>
    <body>
        <div class="container">  <h1>Grant Proposal Evaluation</h1>
            <form method="post">
                <label for="proposal">Project Proposal:</label><br>
                <textarea id="proposal" name="proposal" rows="10" cols="80"></textarea><br><br>
                <button type="submit">Evaluate</button>
            </form>
        </div>
    </body>
    </html>
    """

@app.post("/", response_class=HTMLResponse)
async def evaluate_proposal(request: Request, proposal: str = Form(...)):
    try:
        project_features, similarity_score, error_message, best_match_grant, best_match_features = evaluate_project(proposal, clusters, cluster_labels, labels, grant_descriptions) #Corrected line

        if error_message:
            return error_message

        results_html = f"""
            <div class="container">
                <h2>Evaluation Results</h2>
                <p>Project Features:</p>
                <ul>
        """
        if project_features:
            for category, rating in project_features.items():
                results_html += f"<li>{category}: {rating:.2f}</li>"
        results_html += "</ul>"

        if similarity_score is not None:
            results_html += f"<p>Similarity Score: {similarity_score:.2f}</p>"
        else:
            results_html += "<p>Similarity Score: N/A </p>"

        if best_match_grant: #Added this block
            results_html += f"<p>Best Matching Grant: {best_match_grant}</p>"
            results_html += "<p>Best Matching Grant Features:</p><ul>"
            if best_match_features:
                for cat, rating in best_match_features.items():
                    results_html += f"<li>{cat}: {rating:.2f}</li>"
            results_html += "</ul>"
        results_html += "</div>"

        return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Grant Proposal Evaluation</title>
                <link rel="stylesheet" href="/static/style.css">
            </head>
            <body>
                <div class="container">
                    <h1>Grant Proposal Evaluation</h1>
                    <form method="post">
                        <label for="proposal">Project Proposal:</label><br>
                        <textarea id="proposal" name="proposal" rows="10" cols="80">{proposal}</textarea><br><br>
                        <button type="submit">Evaluate</button>
                    </form>
                    {results_html}
                </div>
            </body>
            </html>
            """

    except Exception as e:
        return f"An error occurred: {e}"