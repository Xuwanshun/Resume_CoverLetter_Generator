import json
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Client with LLM
# -----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# Help Function
# -----------------------------
def load_json(path: str) -> List[Dict[str, Any]]:
    # open file at 'path'; "r": reading mode; store opened file in f 
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f) 
    
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def safe_json_loads(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Could not parse JSON from model output.")
# -----------------------------
# Template style extraction
# -----------------------------
def extract_template_style(template_text: str, doc_type: str) -> Dict[str, Any]:
    """
    Summarize a resume or cover-letter template into style instructions.
    doc_type should be 'resume' or 'cover_letter'.
    """
    schema = {
        "section_order": ["string"],
        "tone": "string",
        "bullet_style": "string or null",
        "length_guidance": "string",
        "format_notes": ["string"]
    }

    prompt = f"""
    You are analyzing a {doc_type} template.

    Return valid JSON only.

    Schema:
    {json.dumps(schema, indent=2)}

    Goal:
    - infer the style and structure from the template text
    - do NOT copy the template content literally unless it is generic structure
    - identify section order, tone, bullet style, approximate length, and formatting notes

    Template text:
    {template_text[:12000]}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": f"You analyze {doc_type} template style."},
            {"role": "user", "content": prompt},
        ],
    )

    return safe_json_loads(response.choices[0].message.content)

def rewrite_project_for_resume(job: Dict[str, Any],project: Dict[str, Any],max_bullets: int = 3) -> Dict[str, Any]:
    schema = {
        "project_id": "string",
        "project_title": "string",
        "display_title": "string",
        "tailored_bullets": ["string"],
        "highlighted_skills": ["string"],
        "matched_requirements": ["string"],
        "best_angle": "string",
        "notes": ["string"]
    }

    prompt = f"""
    You are rewriting one candidate project for a tailored resume.

    Return valid JSON only.

    Schema:
    {json.dumps(schema, indent=2)}

    Rules:
    - Use ONLY facts supported by the project data
    - Do NOT invent metrics, tools, responsibilities, dates, or outcomes
    - Emphasize the parts of the project most relevant to the job
    - It is okay to rephrase and reorder details for stronger alignment
    - Output at most {max_bullets} bullets
    - Bullets should be concise and action-oriented

    Job:
    {json.dumps(job, indent=2)}

    Project:
    {json.dumps(project, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You rewrite project experience into truthful, targeted resume bullets."},
            {"role": "user", "content": prompt},
        ],
    )

    return safe_json_loads(response.choices[0].message.content)

def generate_project_experience_section(job: Dict[str, Any],rewritten_projects: List[Dict[str, Any]],resume_template_text: str) -> str:
    style_profile = extract_template_style(resume_template_text, doc_type="resume")

    prompt = f"""
    Generate only the PROJECT / EXPERIENCE section of a tailored resume in plain text.

    Rules:
    - Use the template only as a style and formatting reference
    - Include only the rewritten project information provided
    - Do NOT invent details
    - Keep strong alignment with the job requirements
    - Make the section look polished and resume-ready
    - Output plain text only
    - Do not generate education, publications, or general summary sections

    Style profile:
    {json.dumps(style_profile, indent=2)}

    Job:
    {json.dumps(job, indent=2)}

    Rewritten projects:
    {json.dumps(rewritten_projects, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.25,
        messages=[
            {"role": "system", "content": "You generate polished project experience sections for resumes."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()


def generate_cover_letter_project_body(job: Dict[str, Any],rewritten_projects: List[Dict[str, Any]],self_intro: str,cover_template_text: str) -> str:
    style_profile = extract_template_style(cover_template_text, doc_type="cover_letter")

    prompt = f"""
    Generate a tailored cover letter in plain text.

    Important:
    - Focus mainly on the candidate's selected project experience
    - Education and publications will be handled separately, so do not rely on them
    - Use the template only as a style/structure reference

    Rules:
    - Use ONLY supported facts from the rewritten projects and self introduction
    - Do NOT invent metrics, technologies, achievements, or responsibilities
    - Explain why the selected projects make the candidate a strong fit
    - Be specific and professional, not generic
    - Output plain text only

    Style profile:
    {json.dumps(style_profile, indent=2)}

    Job:
    {json.dumps(job, indent=2)}

    Rewritten projects:
    {json.dumps(rewritten_projects, indent=2)}

    Self introduction:
    {self_intro}
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0.35,
        messages=[
            {"role": "system", "content": "You write tailored cover letters grounded in project experience."},
            {"role": "user", "content": prompt},
        ],
    )

    return response.choices[0].message.content.strip()

def validate_project_text(generated_text: str,job: Dict[str, Any],selected_projects: List[Dict[str, Any]],rewritten_projects: List[Dict[str, Any]],doc_type: str) -> Dict[str, Any]:
    schema = {
        "is_fully_supported": "boolean",
        "issues": ["string"],
        "corrected_text": "string"
    }

    prompt = f"""
    You are validating a generated {doc_type}.

    Return valid JSON only.

    Schema:
    {json.dumps(schema, indent=2)}

    Rules:
    - Check whether all important claims are supported by the source project data
    - Flag invented tools, metrics, responsibilities, dates, or outcomes
    - If needed, correct the text while keeping it strong and professional
    - If already valid, return it unchanged

    Job:
    {json.dumps(job, indent=2)}

    Original selected projects:
    {json.dumps(selected_projects, indent=2)}

    Rewritten projects:
    {json.dumps(rewritten_projects, indent=2)}

    Generated {doc_type}:
    {generated_text}
    """

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": f"You verify generated {doc_type}s against project source data."},
            {"role": "user", "content": prompt},
        ],
    )

    return safe_json_loads(response.choices[0].message.content)

def rewrite_projects_for_resume(job: Dict[str, Any],selected_projects: List[Dict[str, Any]],max_bullets_per_project: int = 3) -> List[Dict[str, Any]]:
    rewritten = []
    for project in selected_projects:
        rewritten.append(
            rewrite_project_for_resume(
                job=job,
                project=project,
                max_bullets=max_bullets_per_project
            )
        )
    return rewritten

def select_top_5_projects(job: Dict[str, Any],projects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Select top 5 projects using:
    1) mandatory_requirements
    2) job_description
    3) optional_requirements
    """

    schema = {
        "selected_project_ids": ["string"],
        "project_rankings": [
            {
                "project_id": "string",
                "rank": "integer",
                "reason": "string",
                "mandatory_match": "strong|medium|weak|none",
                "job_description_match": "strong|medium|weak|none",
                "optional_match": "strong|medium|weak|none",
                "best_angle": "string"
            }
        ],
        "mandatory_coverage": [
            {
                "requirement": "string",
                "covered_by_project_ids": ["string"],
                "coverage_strength": "strong|medium|weak|uncovered"
            }
        ],
        "notes": ["string"]
    }

    prompt = f"""
    You are selecting the top 5 best candidate projects for a tailored resume and cover letter.

    Return valid JSON only.

    Schema:
    {json.dumps(schema, indent=2)}

    Selection objective:
    - Select exactly 5 projects if at least 5 good candidates exist
    - Otherwise select fewer if there are not 5 meaningful matches
    - Prioritize projects that collectively cover the mandatory requirements
    - Use the job description to understand the real work, emphasis, and context
    - Use optional requirements only as a lower-priority bonus signal
    - Avoid redundant projects unless redundancy is necessary to cover the mandatory requirements well

    Priority order:
    1. mandatory_requirements
    2. job_description
    3. optional_requirements

    Detailed rules:
    - Mandatory requirements are the highest-priority signal
    - Job description should be used to understand which project details matter most
    - Optional requirements should only help break ties or strengthen borderline choices
    - Prefer projects with direct and well-supported evidence
    - Use ONLY the provided project data
    - Do NOT invent project skills, metrics, outcomes, or responsibilities
    - The selected 5 should work well together as a set, not just as isolated strong projects

    How to rank:
    - rank 1 = strongest overall value for this application
    - rank 5 = still useful, but lower priority than the others

    Job:
    {json.dumps(job, indent=2)}

    Projects:
    {json.dumps(projects, indent=2)}
    """

    response = client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You select candidate projects by prioritizing mandatory requirements, then job-description fit, then optional requirements."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    result = safe_json_loads(response.choices[0].message.content)

    valid_ids = {p.get("id") for p in projects if p.get("id")}
    selected_ids = [pid for pid in result.get("selected_project_ids", []) if pid in valid_ids]

    # keep at most 5
    result["selected_project_ids"] = selected_ids[:5]

    # keep only ranking items for valid IDs
    rankings = []
    seen = set()
    for item in result.get("project_rankings", []):
        pid = item.get("project_id")
        if pid in valid_ids and pid not in seen:
            rankings.append(item)
            seen.add(pid)

    result["project_rankings"] = rankings

    return result


def attach_selected_projects(
    selection_result: Dict[str, Any],
    projects: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    by_id = {p.get("id"): p for p in projects if p.get("id")}
    ranking_by_id = {
        item["project_id"]: item
        for item in selection_result.get("project_rankings", [])
        if item.get("project_id")
    }

    selected_projects = []
    for pid in selection_result.get("selected_project_ids", []):
        if pid in by_id:
            project = dict(by_id[pid])
            project["_selection_info"] = ranking_by_id.get(pid, {})
            selected_projects.append(project)

    return selected_projects

def generate_project_package(
    job: Dict[str, Any],
    selected_projects: List[Dict[str, Any]],
    self_intro: str,
    resume_template_text: str,
    cover_template_text: str
) -> Dict[str, Any]:
    rewritten_projects = rewrite_projects_for_resume(
        job=job,
        selected_projects=selected_projects,
        max_bullets_per_project=3
    )

    resume_project_section_raw = generate_project_experience_section(
        job=job,
        rewritten_projects=rewritten_projects,
        resume_template_text=resume_template_text
    )

    cover_letter_raw = generate_cover_letter_project_body(
        job=job,
        rewritten_projects=rewritten_projects,
        self_intro=self_intro,
        cover_template_text=cover_template_text
    )

    resume_validation = validate_project_text(
        generated_text=resume_project_section_raw,
        job=job,
        selected_projects=selected_projects,
        rewritten_projects=rewritten_projects,
        doc_type="resume project section"
    )

    cover_validation = validate_project_text(
        generated_text=cover_letter_raw,
        job=job,
        selected_projects=selected_projects,
        rewritten_projects=rewritten_projects,
        doc_type="cover letter"
    )

    return {
        "rewritten_projects": rewritten_projects,
        "resume_project_section_raw": resume_project_section_raw,
        "resume_project_section_final": resume_validation["corrected_text"],
        "resume_project_section_validation": resume_validation,
        "cover_letter_raw": cover_letter_raw,
        "cover_letter_final": cover_validation["corrected_text"],
        "cover_letter_validation": cover_validation
    }