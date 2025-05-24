import re
import io
import logging
import PyPDF2
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load SpaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    logger.warning("Downloading SpaCy model...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load('en_core_web_sm')

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        
        # Clean up text
        text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
        text = text.strip()
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def preprocess_text(text):
    """
    Preprocess text using SpaCy for tokenization, stopword removal, and lemmatization
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    try:
        # Create SpaCy Doc object
        doc = nlp(text.lower())
        
        # Get lemmatized tokens, excluding stopwords and punctuation
        processed_tokens = [token.lemma_ for token in doc 
                           if not token.is_stop 
                           and not token.is_punct
                           and not token.is_space
                           and len(token.text) > 1]
        
        # Join tokens back into a single string
        processed_text = ' '.join(processed_tokens)
        
        return processed_text
    except Exception as e:
        logger.error(f"Error preprocessing text: {str(e)}")
        return text

def extract_key_skills(text, job_text):
    """
    Extract key skills from resume based on job description
    
    Args:
        text (str): Resume text
        job_text (str): Job description text
        
    Returns:
        list: List of key skills found in both job and resume
    """
    # Create SpaCy Doc objects
    job_doc = nlp(job_text.lower())
    resume_doc = nlp(text.lower())
    
    # Extract noun chunks and named entities as potential skills
    job_skills = set()
    for chunk in job_doc.noun_chunks:
        # Only consider chunks that are 1-3 words long
        if 1 <= len(chunk.text.split()) <= 3:
            job_skills.add(chunk.text)
    
    # Add named entities
    for ent in job_doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'GPE']:
            job_skills.add(ent.text)
    
    # Match skills in resume
    found_skills = []
    resume_text_lower = text.lower()
    
    for skill in job_skills:
        if skill in resume_text_lower:
            found_skills.append(skill)
    
    return sorted(found_skills)

def rank_resumes(resume_data, preprocessed_job_desc):
    """
    Rank resumes based on similarity to job description
    
    Args:
        resume_data (list): List of dictionaries containing resume information
        preprocessed_job_desc (str): Preprocessed job description text
        
    Returns:
        list: List of dictionaries with resume data and ranking scores
    """
    if not resume_data:
        return []
    
    # Extract preprocessed texts
    preprocessed_texts = [resume['preprocessed_text'] for resume in resume_data]
    
    # Add job description to the texts for vectorization
    all_texts = preprocessed_texts + [preprocessed_job_desc]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Get job description vector (last item in the matrix)
    job_vec = tfidf_matrix[-1]
    
    # Calculate cosine similarity between each resume and job description
    similarities = cosine_similarity(tfidf_matrix[:-1], job_vec)
    
    # Create result list with scores
    ranked_resumes = []
    for i, resume in enumerate(resume_data):
        score = float(similarities[i][0])  # Get similarity score
        key_skills = extract_key_skills(resume['text'], preprocessed_job_desc)
        
        # Store original text for resume creation
        ranked_resumes.append({
            'filename': resume['filename'],
            'score': score,
            'match_percentage': round(score * 100, 2),
            'key_skills': key_skills,
            'skill_count': len(key_skills),
            'text': resume['text'],
            'is_ats_friendly': is_ats_friendly(resume['text'], key_skills)
        })
    
    # Sort by score (descending)
    ranked_resumes = sorted(ranked_resumes, key=lambda x: x['score'], reverse=True)
    
    # Add rank
    for i, resume in enumerate(ranked_resumes):
        resume['rank'] = i + 1
    
    return ranked_resumes

def is_ats_friendly(resume_text, skills):
    """
    Check if a resume is ATS friendly
    
    Args:
        resume_text (str): The resume text
        skills (list): List of identified skills
        
    Returns:
        bool: True if the resume is considered ATS friendly
    """
    # Simple heuristic for ATS friendliness
    score = 0
    
    # Check if has a good number of identified skills (at least 5)
    if len(skills) >= 5:
        score += 1
        
    # Check if resume contains common section headers
    headers = ['experience', 'education', 'skills', 'work history', 'projects']
    for header in headers:
        if header in resume_text.lower():
            score += 0.5
            
    # Check if text is well-structured (ratio of newlines to text length)
    newline_count = resume_text.count('\n')
    if 0.01 <= newline_count / len(resume_text) <= 0.1:  # Good spacing ratio
        score += 1
        
    # Check for presence of contact information
    contact_patterns = [
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number
        r'linkedin\.com\/in\/[a-zA-Z0-9-]+'  # LinkedIn
    ]
    
    for pattern in contact_patterns:
        if re.search(pattern, resume_text):
            score += 0.5
            
    return score >= 2.5  # Threshold for ATS friendliness

def filter_resumes_by_rank(ranked_resumes, min_rank=1, max_rank=None):
    """
    Filter resumes by rank range
    
    Args:
        ranked_resumes (list): List of ranked resumes
        min_rank (int): Minimum rank to include (inclusive)
        max_rank (int): Maximum rank to include (inclusive), None for no upper limit
        
    Returns:
        list: Filtered list of resumes
    """
    if not ranked_resumes:
        return []
        
    filtered_resumes = [r for r in ranked_resumes if r['rank'] >= min_rank]
    
    if max_rank is not None:
        filtered_resumes = [r for r in filtered_resumes if r['rank'] <= max_rank]
        
    return filtered_resumes

def generate_ats_friendly_resume(resume_text, job_description, skills):
    """
    Generate an ATS-friendly version of a resume based on job description and skills
    
    Args:
        resume_text (str): Original resume text
        job_description (str): Job description text
        skills (list): List of key skills identified
        
    Returns:
        str: ATS-friendly resume text
    """
    # Extract sections from the original resume
    sections = extract_resume_sections(resume_text)
    
    # Build the enhanced resume
    enhanced_resume = []
    
    # 1. Add a professional summary focused on matching the job
    enhanced_resume.append("# PROFESSIONAL SUMMARY")
    
    # Create a summary using key skills from the job description
    skill_summary = "Experienced professional with expertise in " + ", ".join(skills[:5])
    if len(skills) > 5:
        skill_summary += f" and {len(skills)-5} other relevant skills"
    skill_summary += "."
    
    enhanced_resume.append(skill_summary)
    enhanced_resume.append("")
    
    # 2. Add highlighted skills section
    enhanced_resume.append("# KEY SKILLS")
    skill_rows = []
    for i in range(0, len(skills), 3):
        row = skills[i:i+3]
        skill_rows.append(" | ".join(row))
    
    enhanced_resume.extend(skill_rows)
    enhanced_resume.append("")
    
    # 3. Add other sections from original resume
    for section_name, section_content in sections.items():
        if section_name.lower() not in ['summary', 'skills'] and section_content.strip():
            enhanced_resume.append(f"# {section_name.upper()}")
            enhanced_resume.append(section_content)
            enhanced_resume.append("")
    
    return "\n".join(enhanced_resume)

def extract_resume_sections(resume_text):
    """
    Extract sections from a resume text
    
    Args:
        resume_text (str): Original resume text
        
    Returns:
        dict: Dictionary with section names as keys and content as values
    """
    # Common section headers in resumes
    section_headers = [
        'summary', 'profile', 'objective', 
        'experience', 'work experience', 'employment', 'work history',
        'education', 'academic background', 'qualifications',
        'skills', 'technical skills', 'competencies', 'expertise',
        'projects', 'professional projects', 'portfolio',
        'certifications', 'certificates', 'licenses',
        'awards', 'honors', 'achievements',
        'publications', 'research', 'papers',
        'languages', 'language proficiency',
        'interests', 'hobbies', 'activities',
        'references', 'professional references'
    ]
    
    # Create a regex pattern to find section headers
    pattern = r'(?i)(?:^|\n)(?:[\d\.\s]*)({})[:\s]*(?:\n|$)'.format('|'.join(section_headers))
    
    # Find all potential section headers
    matches = list(re.finditer(pattern, resume_text))
    
    sections = {}
    
    # Process each section
    for i, match in enumerate(matches):
        section_name = match.group(1).strip()
        start_pos = match.end()
        
        # If this is the last section, the end is the end of the text
        if i == len(matches) - 1:
            end_pos = len(resume_text)
        else:
            end_pos = matches[i+1].start()
        
        # Extract the section content
        section_content = resume_text[start_pos:end_pos].strip()
        sections[section_name] = section_content
    
    # If no sections were found, create a single "content" section
    if not sections:
        sections['content'] = resume_text
    
    return sections

def create_resume_template():
    """
    Create a basic resume template
    
    Returns:
        str: Basic resume template
    """
    template = [
        "# CONTACT INFORMATION",
        "Full Name: [Your Name]",
        "Email: [Your Email]",
        "Phone: [Your Phone Number]",
        "LinkedIn: [Your LinkedIn Profile]",
        "",
        "# PROFESSIONAL SUMMARY",
        "[A brief summary highlighting your key qualifications and career focus]",
        "",
        "# SKILLS",
        "• [Skill 1]",
        "• [Skill 2]",
        "• [Skill 3]",
        "",
        "# WORK EXPERIENCE",
        "[Job Title] | [Company Name] | [Start Date - End Date]",
        "• [Accomplishment or responsibility]",
        "• [Accomplishment or responsibility]",
        "",
        "# EDUCATION",
        "[Degree] in [Field of Study] | [University Name] | [Graduation Year]",
        "",
        "# CERTIFICATIONS",
        "• [Certification Name] | [Issuing Organization] | [Year]",
        "",
        "# PROJECTS",
        "[Project Name]",
        "• [Description of the project and your role]",
        "",
    ]
    
    return "\n".join(template)
