import os
import requests
import json
from typing import Any, Dict
from bs4 import BeautifulSoup
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# download html text and convert html to string
def fetch(url):
    headers = {"User-Agent": "job-post-fetcher/1.0"}
    response = requests.get(url, headers=headers, timeout=20) # send request to web
    response.raise_for_status() # if error, return error number
    return response.text

def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parse") #"html.parse": Python’s built-in HTML parser

    # Remove noisy tags
    for tag in soup(["script", "style", "noscript", "svg", "img", "iframe", "footer", "nav"]):
        tag.decompose() #removes that tag and everything inside it from tree
    
    # Try to keep the main content if possible
    main = soup.find("main")
    if main:
        return str(main)
    
    return str(soup)

def extract_info(clean_web: str) -> str:
    soup = BeautifulSoup(clean_web, "html.parser")
    text = soup.get_text("\n", strip=True) # removes the HTML tags and returns only the text content.
    lines = [line.strip() for line in text.splitlines()] # This splits the text into a list
    lines = [line for line in lines if line] # remove empty
    text = "\n".join(lines)
    return text

def LLM_info_extration(job_text: str) -> dict[str, any]:
    schema_description = {
    "job_title": "string or null",
    "company_name": "string or null",
    "company_description": "string or null",
    "location": "string or null",
    "employment_type": "string or null",
    "job_role_description": "string or null",
    "role_responsibilities": ["string"],
    "requirements": ["string"],
    "optional_requirements": ["string"],
    "education": ["string"],
    "years_of_experience": "string or null",
    "skills": ["string"],
    "summary": "string or null"
    }

    prompt = f"""
    You are a job-posting parser.

    Extract the job information from the text below and return valid JSON only.
    Do not include markdown fences.
    Do not include explanations.

    Schema:
    {json.dumps(schema_description, indent=2)}

    Notes:
    - "company_description" = brief description of what the company does
    - "job_role_description" = short overview of the role itself
    - "role_responsibilities" = main things the person will do
    - "requirements" = required qualifications only
    - "optional_requirements" = preferred / nice-to-have / bonus qualifications

    Job text:
    {job_text[:12000]}
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You extract structured data from job postings."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    return json.loads(content)

def parse_job(url: str) -> Dict[str, Any]:
    html = fetch(url)
    cleaned_html = clean_html(html)
    job_text = extract_info(cleaned_html)
    structured_data = parse_job(job_text)

    return {
        "url": url,
        "job_text": job_text,
        "structured_data": structured_data,
    }
        
