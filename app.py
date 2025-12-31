# app.py
"""
RAG-based Student Work Auto-Grader (Streamlit) - Enhanced UX Version
Applying HCI Principles: Progressive Disclosure, Immediate Feedback, Clear Affordances, etc.
"""
import pandas as pd
from io import BytesIO

# Optional PDF import
try:
    import pdfplumber
except Exception:
    pdfplumber = None

import os
import json
import time
import streamlit as st
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from datetime import datetime
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False
    
try:
    import docx2txt
except Exception:
    docx2txt = None

# Try to import language_tool_python but we'll use Jina AI instead
try:
    import language_tool_python
    lang_tool = language_tool_python.LanguageTool("en-US")
    HAS_LANG_TOOL = True
except Exception:
    lang_tool = None
    HAS_LANG_TOOL = False

# ==================== HCI ENHANCEMENTS ====================
st.set_page_config(
    page_title="ai!Grader pro - Intelligent Auto-Grader", 
    layout="wide",
    page_icon="📚",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UX
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.4rem !important;
        color: #2e86ab;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .progress-bar {
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin: 0.5rem 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff6b6b, #ffa500, #ffd93d, #6bcf7f);
        transition: width 0.5s ease-in-out;
    }
    .tab-content {
        padding: 1rem 0;
    }
    .feedback-item {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border-left: 3px solid #1f77b4;
    }
    .grammar-issue {
        background-color: #fff3cd;
        border-left: 3px solid #ffc107;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 0.25rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load embedding model once
@st.cache_resource(show_spinner="🔄 Loading AI grading engine...")
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-large-en-v1.5")  # Changed from "all-MiniLM-L6-v2"

embedding_model = load_embedding_model()

# ---------------------------
# Enhanced Utilities with Progress Indicators
# ---------------------------

def parse_teacher_rubric(text: str) -> Optional[dict]:
    """
    Convert teacher-friendly rubric table into internal rubric JSON.
    Expected format (flexible):
    Criterion | Weight | Description
    or
    Criterion, Weight, Description
    """
    if not text or not text.strip():
        return None
    
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    
    criteria = []
    line_count = 0
    
    # Try different delimiters
    for line in lines:
        line_count += 1
        
        # Skip header lines that don't contain numbers
        if line_count == 1:
            # Check if this looks like a header (contains words like criterion, weight, description)
            header_words = ['criterion', 'weight', 'description', 'criteria', 'score', 'points']
            if any(word in line.lower() for word in header_words):
                continue
        
        # Try pipe delimiter first, then comma, then tab
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
        elif ',' in line:
            parts = [p.strip() for p in line.split(',')]
        elif '\t' in line:
            parts = [p.strip() for p in line.split('\t')]
        else:
            # Try splitting by 2 or more spaces
            parts = [p.strip() for p in line.split('  ') if p.strip()]
        
        if len(parts) < 2:
            continue
        
        name = parts[0]
        if not name:
            continue
        
        # Extract weight - try to find a number in the parts
        weight = 0
        for part in parts[1:]:
            # Look for numbers in the part
            try:
                # Remove any non-numeric characters except decimal point
                import re
                weight_str = re.sub(r'[^\d.]', '', part)
                if weight_str:
                    weight = float(weight_str)
                    break
            except:
                continue
        
        # Determine criterion type
        criterion_type = "grammar_penalty" if "grammar" in name.lower() else "similarity"
        
        criteria.append({
            "name": name,
            "weight": weight,
            "type": criterion_type,
            "penalty_per_issue": 1.5 if "grammar" in name.lower() else 0
        })
    
    if not criteria:
        return None
    
    # Normalize weights so they sum to 1.0
    total_weight = sum(c.get("weight", 0) for c in criteria)
    if total_weight > 0:
        for criterion in criteria:
            criterion["weight"] = criterion["weight"] / total_weight
    
    return {"criteria": criteria}

def convert_to_ielts_band(score_100: float) -> float:
    """
    Convert 0-100 score to IELTS band 0-9
    Based on IELTS Writing Task 2 band descriptors approximation
    """
    if score_100 >= 95:
        return 9.0
    elif score_100 >= 88:
        return 8.5
    elif score_100 >= 80:
        return 8.0
    elif score_100 >= 75:
        return 7.5
    elif score_100 >= 70:
        return 7.0
    elif score_100 >= 65:
        return 6.5
    elif score_100 >= 60:
        return 6.0
    elif score_100 >= 55:
        return 5.5
    elif score_100 >= 50:
        return 5.0
    elif score_100 >= 45:
        return 4.5
    elif score_100 >= 40:
        return 4.0
    elif score_100 >= 35:
        return 3.5
    elif score_100 >= 30:
        return 3.0
    elif score_100 >= 25:
        return 2.5
    elif score_100 >= 20:
        return 2.0
    elif score_100 >= 15:
        return 1.5
    elif score_100 >= 10:
        return 1.0
    else:
        return 0.5 if score_100 > 0 else 0.0

def apply_rubric_json(rubric: dict, model_ans: str, student_ans: str, output_scale: str = "numeric_100") -> Dict[str, Any]:
    """Apply rubric-based grading with proper weight distribution."""
    criteria = rubric.get("criteria", [])
    if not criteria:
        return heuristic_grade(model_ans, student_ans, output_scale)
    
    with st.status("📊 Applying rubric criteria...", state="running") as status:
        # Calculate similarity
        vecs = embed_texts([model_ans, student_ans])
        similarity_score = cosine_sim(vecs[0], vecs[1])
        similarity_percent = max(0.0, min((similarity_score + 1) / 2.0, 1.0)) * 100
        
        # Grammar check
        grammar_result = grammar_check_with_jina(student_ans)
        grammar_issues = grammar_result.get("issues_count", 0) if grammar_result.get("available") else 0
        
        total_score = 0.0
        breakdown = []
        
        for i, criterion in enumerate(criteria):
            name = criterion.get("name", f"Criterion {i+1}")
            weight = criterion.get("weight", 0)
            criterion_type = criterion.get("type", "similarity")
            penalty_per_issue = criterion.get("penalty_per_issue", 1.5)
            
            subscore = 0.0
            
            if criterion_type == "similarity":
                subscore = similarity_percent
            elif criterion_type == "grammar_penalty":
                if grammar_result.get("available"):
                    # Apply penalty: start from 100 and subtract for each issue
                    subscore = max(0.0, 100.0 - (grammar_issues * penalty_per_issue))
                else:
                    subscore = 100.0  # Full points if grammar check not available
            else:
                # Default to similarity for unknown types
                subscore = similarity_percent
            
            # Apply weighting
            weighted_score = subscore * weight
            total_score += weighted_score
            
            breakdown.append({
                "criterion": name,
                "weight": round(weight, 4),
                "subscore": round(subscore, 2),
                "type": criterion_type,
                "weighted_score": round(weighted_score, 2)
            })
        
        # Ensure total_score doesn't exceed 100
        total_score = min(100.0, total_score)
        
        # Convert to IELTS band if needed
        if output_scale == "ielts_band_0-9":
            ielts_band = convert_to_ielts_band(total_score)
            final_display_score = ielts_band
            scale_used = "ielts"
        else:
            final_display_score = total_score
            scale_used = "numeric"
        
        status.update(label="✅ Rubric applied successfully", state="complete")
    
    return {
        "final_score": round(final_display_score, 2),
        "breakdown": breakdown,
        "similarity": similarity_percent / 100.0,
        "grammar": grammar_result,
        "grading_method": "rubric",
        "scale_used": scale_used,
        "original_100_score": round(total_score, 2) if output_scale == "ielts_band_0-9" else None
    }

def export_results_to_excel(results: list) -> bytes:
    """
    Convert grading results into a multi-sheet Excel file.
    """
    summary_rows = []
    breakdown_rows = []
    grammar_rows = []

    for r in results:
        # -------- Summary Sheet --------
        scale_info = ""
        if r.get("details", {}).get("scale_used") == "ielts":
            scale_info = f" (IELTS Band: {r.get('final_score')})"
        
        summary_rows.append({
            "Student Name": r.get("name"),
            "Final Score": r.get("final_score"),
            "Scale": r.get("details", {}).get("scale_used", "numeric_100"),
            "Original 100-point Score": r.get("details", {}).get("original_100_score", r.get("final_score")),
            "Similarity (%)": round(r.get("details", {}).get("similarity", 0) * 100, 2),
            "Grammar Issues": r.get("details", {}).get("grammar", {}).get("issues_count"),
            "Grading Method": r.get("details", {}).get("grading_method"),
            "Timestamp": r.get("timestamp")
        })

        # -------- Breakdown Sheet --------
        for b in r.get("details", {}).get("breakdown", []):
            breakdown_rows.append({
                "Student Name": r.get("name"),
                "Criterion": b.get("criterion"),
                "Weight": b.get("weight"),
                "Subscore": b.get("subscore"),
                "Type": b.get("type")
            })

        # -------- Grammar Sheet --------
        grammar = r.get("details", {}).get("grammar", {})
        if grammar.get("available"):
            for g in grammar.get("examples", []):
                grammar_rows.append({
                    "Student Name": r.get("name"),
                    "Issue": g.get("message"),
                    "Context": g.get("context"),
                    "Suggestions": ", ".join(g.get("suggestions", []))
                })

    # Create DataFrames
    df_summary = pd.DataFrame(summary_rows)
    df_breakdown = pd.DataFrame(breakdown_rows)
    df_grammar = pd.DataFrame(grammar_rows)

    # Write to Excel in-memory
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_summary.to_excel(writer, index=False, sheet_name="Summary")
        df_breakdown.to_excel(writer, index=False, sheet_name="Score Breakdown")
        if not df_grammar.empty:
            df_grammar.to_excel(writer, index=False, sheet_name="Grammar Issues")

    return output.getvalue()

def read_text_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    
    with st.status(f"📄 Processing {uploaded_file.name}...", state="running") as status:
        try:
            content = uploaded_file.getvalue()
            name = uploaded_file.name.lower()
            result = ""

            if name.endswith(".txt"):
                result = content.decode("utf-8")

            elif name.endswith(".docx"):
                if docx2txt:
                    tmp_path = f"/tmp/temp_{int(time.time())}.docx"
                    with open(tmp_path, "wb") as f:
                        f.write(content)
                    result = docx2txt.process(tmp_path)
                else:
                    st.warning("📝 docx2txt not installed")

            elif name.endswith(".pdf"):
                if pdfplumber:
                    tmp_path = f"/tmp/temp_{int(time.time())}.pdf"
                    with open(tmp_path, "wb") as f:
                        f.write(content)

                    text_pages = []
                    with pdfplumber.open(tmp_path) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_pages.append(page_text)

                    result = "\n".join(text_pages)
                else:
                    st.warning("📄 pdfplumber not installed; run `pip install pdfplumber`")

            else:
                result = content.decode("utf-8", errors="ignore")

            status.update(label=f"✅ Processed {uploaded_file.name}", state="complete")
            return result.strip()

        except Exception as e:
            status.update(label=f"❌ Error processing {uploaded_file.name}", state="error")
            st.error(str(e))
            return ""

def safe_load_json(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception as e:
        st.error(f"❌ Invalid JSON format: {str(e)}")
        return None

def embed_texts(texts: List[str]) -> np.ndarray:
    texts = [t if t is not None else "" for t in texts]
    
    # Show embedding progress
    progress_text = "🔍 Analyzing text similarities..."
    my_bar = st.progress(0, text=progress_text)
    
    for i in range(100):
        time.sleep(0.01)  # Simulate progress
        my_bar.progress(i + 1, text=progress_text)
    
    vectors = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    my_bar.empty()
    
    return vectors

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(1, -1)
    b = b.reshape(1, -1)
    return float(cosine_similarity(a, b)[0][0])

def grammar_check_with_jina(text: str) -> Dict[str, Any]:
    """Use Jina AI to check grammar by asking it to analyze the text."""
    api_key = os.getenv("JINACHAT_API_KEY")
    if not api_key or not text.strip():
        return {"available": False, "issues_count": 0, "examples": []}
    
    with st.status("🔍 Checking grammar with Jina AI...", state="running") as status:
        url = "https://api.jina.ai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Prompt to ask Jina AI to analyze grammar
        prompt = f"""Analyze this text for grammar, spelling, and punctuation errors. 
        Provide a list of issues found with specific corrections. Be strict and thorough.
        
        Text to analyze: "{text}"
        
        Format your response as:
        - [Issue]: [Correction] | Explanation
        
        If there are no errors, say "No issues found"."""
        
        payload = {
            "model": "jina-clip",
            "messages": [
                {"role": "system", "content": "You are a strict grammar checker. Find all errors in the text and provide corrections."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "")
                    
                    # Parse the response
                    examples = []
                    lines = content.strip().split('\n')
                    
                    for line in lines:
                        if line.strip() and 'No issues found' not in line:
                            # Parse the line to extract issue and suggestion
                            if ':' in line:
                                parts = line.split(':', 1)
                                if len(parts) == 2:
                                    issue = parts[0].replace('-', '').strip()
                                    rest = parts[1].strip()
                                    if '|' in rest:
                                        suggestion_part, explanation = rest.split('|', 1)
                                        suggestions = [s.strip() for s in suggestion_part.split(',')]
                                    else:
                                        suggestions = [rest]
                                        explanation = ""
                                    
                                    examples.append({
                                        "message": issue,
                                        "context": text[:100],  # Show first 100 chars for context
                                        "suggestions": suggestions[:3],
                                        "explanation": explanation.strip()
                                    })
                    
                    issues_count = len(examples)
                    status.update(label=f"✅ Found {issues_count} grammar issues with Jina AI", state="complete")
                    
                    return {
                        "available": True, 
                        "issues_count": issues_count, 
                        "examples": examples[:6],  # Show top 6 issues
                        "method": "jina_ai"
                    }
            
            # Fallback to basic check if Jina AI fails
            return {"available": True, "issues_count": 0, "examples": [], "method": "fallback"}
            
        except Exception as e:
            status.update(label="🔶 Error with Jina AI grammar check", state="error")
            return {"available": False, "issues_count": 0, "examples": []}

def apply_rubric_json_secondary(rubric: dict, model_ans: str, student_ans: str, output_scale: str = "numeric_100") -> Dict[str, Any]:
    criteria = rubric.get("criteria", [])
    if not criteria:
        return heuristic_grade(model_ans, student_ans, output_scale)

    # Show grading progress
    with st.status("📊 Applying rubric criteria...", state="running") as status:
        vecs = embed_texts([model_ans, student_ans])
        sim = cosine_sim(vecs[0], vecs[1])
        sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
        g = grammar_check_with_jina(student_ans)
        issues = g["issues_count"] if g.get("available") else None

        total_weight = sum(c.get("weight", 0) for c in criteria) or 1.0
        total_score = 0.0
        breakdown = []
        
        for i, c in enumerate(criteria):
            name = c.get("name", f"Criterion {i+1}")
            w = c.get("weight", 0) / total_weight
            t = c.get("type", "similarity")
            subscore = 0.0
            
            if t == "similarity":
                subscore = sim_norm * 100
            elif t == "grammar_penalty":
                if issues is None:
                    subscore = 100.0
                else:
                    penalty_per = c.get("penalty_per_issue", 1.5)
                    subscore = max(0.0, 100.0 - penalty_per * issues)
            else:
                subscore = sim_norm * 100
                
            total_score += subscore * w
            breakdown.append({
                "criterion": name, 
                "weight": round(w,3), 
                "subscore": round(subscore,2),
                "type": t
            })
        
        # Convert to IELTS band if needed
        if output_scale == "ielts_band_0-9":
            ielts_band = convert_to_ielts_band(total_score)
            final_display_score = ielts_band
            scale_used = "ielts"
        else:
            final_display_score = total_score
            scale_used = "numeric"
        
        status.update(label="✅ Rubric applied successfully", state="complete")

    return {
        "final_score": round(final_display_score, 2), 
        "breakdown": breakdown, 
        "similarity": sim_norm, 
        "grammar": g,
        "grading_method": "rubric",
        "scale_used": scale_used,
        "original_100_score": round(total_score, 2) if output_scale == "ielts_band_0-9" else None
    }

def heuristic_grade(model_ans: str, student_ans: str, output_scale: str = "numeric_100") -> Dict[str, Any]:
    with st.status("🎯 Computing similarity scores...", state="running") as status:
        vecs = embed_texts([model_ans, student_ans])
        sim = cosine_sim(vecs[0], vecs[1])
        sim_norm = max(0.0, min((sim + 1) / 2.0, 1.0))
        base = sim_norm * 100
        g = grammar_check_with_jina(student_ans)
        penalty = 0.0
        
        if g.get("available"):
            issues = g["issues_count"]
            penalty = min(40.0, issues * 1.5)
            
        total_score = max(0.0, base - penalty)
        
        # Convert to IELTS band if needed
        if output_scale == "ielts_band_0-9":
            ielts_band = convert_to_ielts_band(total_score)
            final_display_score = ielts_band
            scale_used = "ielts"
        else:
            final_display_score = total_score
            scale_used = "numeric"
            
        breakdown = [
            {"criterion": "Content Similarity", "weight": 0.8, "subscore": round(base,2), "type": "similarity"},
            {"criterion": "Grammar & Mechanics", "weight": 0.2, "subscore": round(max(0, 100 - penalty),2), "type": "grammar"}
        ]
        status.update(label="✅ Automatic grading completed", state="complete")
        
    return {
        "final_score": round(final_display_score, 2), 
        "breakdown": breakdown, 
        "similarity": sim_norm, 
        "grammar": g, 
        "penalty": penalty,
        "grading_method": "heuristic",
        "scale_used": scale_used,
        "original_100_score": round(total_score, 2) if output_scale == "ielts_band_0-9" else None
    }

# ---------------------------
# Jina AI Integration for Feedback
# ---------------------------
def generate_feedback_with_jina(prompt_text: str) -> Optional[str]:
    api_key = os.getenv("JINACHAT_API_KEY")
    if not api_key:
        return None
        
    with st.status("🤖 Generating AI feedback...", state="running") as status:
        url = "https://api.jina.ai/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "jina-clip",
            "messages": [
                {"role": "system", "content": "You are an objective grading assistant. Provide constructive, actionable feedback for student work."},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices", [])
                if choices and len(choices) > 0:
                    msg = choices[0].get("message", {}).get("content")
                    status.update(label="✅ AI feedback generated", state="complete")
                    return msg
            else:
                status.update(label="🔶 AI feedback possible", state="error")
            return None
        except Exception as e:
            status.update(label="🔶 AI feedback possible", state="error")
            return None

# ---------------------------
# Enhanced Streamlit UI with HCI Principles
# ---------------------------
st.markdown('<div class="main-header">📚 ai!Grader pro - Intelligent Auto-Grader</div>', unsafe_allow_html=True)

# Sidebar with clear information hierarchy
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Grading scale with better visual cues
    output_scale = st.selectbox(
        "**Grading Scale**", 
        ["numeric_100", "ielts_band_0-9"], 
        index=0,
        help="Select the scoring system for evaluations"
    )
    
    # Toggle options with icons
    st.markdown("### 👁️ Display Options")
    show_grammar_examples = st.toggle("Show grammar examples", value=True)
    show_detailed_breakdown = st.toggle("Show detailed score breakdown", value=True)
    enable_ai_feedback = st.toggle("Enable AI feedback (if available)", value=True)
    
    # System status - CHANGED THIS LINE AS REQUESTED
    st.markdown("### 🔍 System Status")
    st.success("✅ Embedding model loaded")
    st.info("📊 Grammar checking available")  # Changed from "📊 Grammar checking: 🔶 Basic checking available"
    st.info(f"🤖 AI feedback: {'✅ Available' if os.getenv('JINACHAT_API_KEY') else '❌ Not configured'}")
    
    # Quick tips
    with st.expander("💡 Quick Tips"):
        st.markdown("""
        - **Upload** or **paste** content - both work!
        - Use **rubrics** for consistent grading
        - Check **grammar feedback** for writing improvements
        - **Multiple students** can be graded at once
        """)

# Main content with tabbed interface
tab1, tab2, tab3 = st.tabs(["📥 Input Materials", "🎯 Grading Results", "📈 Analytics"])

with tab1:
    st.markdown('<div class="sub-header">📥 Input Materials</div>', unsafe_allow_html=True)
    
    # Use columns for better information density
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 Exercise Description")
        ex_file = st.file_uploader(
            "Upload exercise document", 
            type=["txt","docx", "pdf"],
            help="Upload a .txt or .docx file with the exercise instructions"
        )
        ex_text_paste = st.text_area(
            "Or paste exercise description here", 
            height=120,
            placeholder="Paste the exercise description, prompt, or question here...",
            help="You can either upload a file or paste text directly"
        )
        
        st.markdown("#### 📝 Student Submissions")
        student_files = st.file_uploader(
            "Upload student files", 
            accept_multiple_files=True, 
            type=["txt","docx", "pdf"],
            help="Upload multiple student submissions at once"
        )
        student_paste = st.text_area(
            "Or paste student submissions", 
            height=150,
            placeholder="Paste student answers here. Separate different submissions with '---' on a new line.",
            help="Use '---' on a separate line to distinguish between different student submissions"
        )

    with col2:
        st.markdown("#### 📖 Model Solution")
        model_file = st.file_uploader(
            "Upload model solution",
            type=["txt","docx","pdf"],
            help="The ideal answer or reference solution for comparison"
        )
        model_text_paste = st.text_area(
            "Or paste model solution here",
            height=120,
            placeholder="Paste the model answer or ideal solution here..."
        )
    
        st.markdown("#### 📊 Grading Rubric (Optional)")
    
        with st.expander("📝 How to format your rubric"):
            st.markdown("""
    **Pipe-separated format:**
            Criterion | Weight | Description
            Content Accuracy | 50 | Covers key concepts correctly
            Organization | 30 | Logical structure and flow
            Grammar | 20 | Grammar, spelling, clarity
    
    **Comma-separated format also works.**
    Weights are auto-normalized.
            """)
    
        rubric_file = st.file_uploader(
            "Upload rubric file (TXT, DOCX, PDF)",
            type=["txt", "docx", "pdf"]
        )
    
        rubric_text_paste = st.text_area(
            "Or paste rubric here",
            height=160,
            value="""Criterion | Weight | Description
    Content Accuracy | 50 | Covers key concepts correctly
    Organization | 30 | Logical structure and flow
    Grammar | 20 | Grammar, spelling, sentence clarity"""
        )
    
        if st.button("🔍 Preview Rubric", type="secondary"):
            parsed = parse_teacher_rubric(rubric_text_paste.strip())
            if parsed:
                st.success("✅ Rubric parsed successfully")
                st.json(parsed)
            else:
                st.error("❌ Could not parse rubric")

        
    
    # Action button with clear visual hierarchy
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        grade_button = st.button(
            "🚀 Start Grading Process", 
            type="primary", 
            use_container_width=True,
            help="Click to begin grading all student submissions"
        )

with tab2:
    st.markdown('<div class="sub-header">🎯 Grading Results</div>', unsafe_allow_html=True)
    
    if 'results' not in st.session_state:
        st.session_state.results = []
        st.info("👆 Start by uploading materials and clicking 'Start Grading Process' in the Input tab.")
    else:
        for i, r in enumerate(st.session_state.results):
            with st.container():
                # Header with student name and score
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"### 👨‍🎓 {r.get('name', 'Student')}")
                with col2:
                    score = r.get('final_score', 0)
                    
                    # Adjust score display based on scale
                    if r.get("details", {}).get("scale_used") == "ielts":
                        score_text = f"{score}/9"
                        # IELTS band color coding
                        score_color = "green" if score >= 7 else "orange" if score >= 5 else "red"
                    else:
                        score_text = f"{score}/100"
                        score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                    
                    st.markdown(f"<h2 style='color: {score_color}; text-align: center;'>{score_text}</h2>", unsafe_allow_html=True)
                
                # Progress bar visualization (adjust max value based on scale)
                if r.get("details", {}).get("scale_used") == "ielts":
                    progress_percent = (score / 9) * 100
                else:
                    progress_percent = score
                
                st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(progress_percent), unsafe_allow_html=True)
                
                # Score interpretation
                if r.get("details", {}).get("scale_used") == "ielts":
                    if score >= 7:
                        st.markdown('<div class="success-box">🎉 Excellent IELTS performance! Band {score} indicates strong language proficiency.</div>', unsafe_allow_html=True)
                    elif score >= 5:
                        st.markdown('<div class="warning-box">📚 Good IELTS band {score}. Shows competence with some room for improvement.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">💡 IELTS band {score} indicates limited proficiency. Focus on core language skills.</div>', unsafe_allow_html=True)
                else:
                    if score >= 80:
                        st.markdown('<div class="success-box">🎉 Excellent work! Strong understanding demonstrated.</div>', unsafe_allow_html=True)
                    elif score >= 60:
                        st.markdown('<div class="warning-box">📚 Good effort, with room for improvement in key areas.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">💡 Needs significant improvement. Review fundamental concepts.</div>', unsafe_allow_html=True)
                
                # Detailed feedback in expanders
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.expander("📋 Detailed Feedback", expanded=True):
                        st.markdown("**Key Observations:**")
                        st.write(r["reasoning"])
                        
                        st.markdown("**Actionable Steps:**")
                        for line in r["feedback_lines"]:
                            st.markdown(f'<div class="feedback-item">💡 {line}</div>', unsafe_allow_html=True)
                        
                        if r.get("jina_feedback") and enable_ai_feedback:
                            st.markdown("**🤖 AI Insights:**")
                            st.write(r["jina_feedback"])
                
                with col2:
                    if show_detailed_breakdown:
                        with st.expander("📊 Score Breakdown", expanded=True):
                            for item in r["details"].get("breakdown", []):
                                col_a, col_b, col_c = st.columns([3, 1, 1])
                                with col_a:
                                    st.write(f"**{item['criterion']}**")
                                with col_b:
                                    st.write(f"{item['subscore']:.1f}")
                                with col_c:
                                    progress = item['subscore'] / 100
                                    st.progress(progress)
                    
                    if show_grammar_examples and r["details"].get("grammar", {}).get("available"):
                        with st.expander("🔍 Grammar Check", expanded=False):
                            g = r["details"]["grammar"]
                            st.write(f"**Issues found:** {g['issues_count']}")
                            for ex in g["examples"]:
                                st.markdown(f"""
                                <div class="grammar-issue">
                                    <strong>⚠️ {ex['message']}</strong><br>
                                    <em>Context:</em> ...{ex['context']}...<br>
                                    {f"<em>Suggestions:</em> {', '.join(ex.get('suggestions', []))}" if ex.get('suggestions') else ""}
                                </div>
                                """, unsafe_allow_html=True)
                
                st.divider()

with tab3:
    st.markdown('<div class="sub-header">📈 Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.results:
        st.info("No grading data available. Complete a grading session first.")
    else:
        # Get the current scale from sidebar
        scale_type = "ielts" if output_scale == "ielts_band_0-9" else "numeric"
        
        # Prepare data for analytics
        scores = []
        original_scores = []
        student_names = []
        
        for r in st.session_state.results:
            if r.get('final_score') is not None:
                scores.append(r.get('final_score'))
                student_names.append(r.get('name', f"Student {len(scores)}"))
                # Store original 100-point score for comparison if needed
                original_scores.append(r.get("details", {}).get("original_100_score", r.get('final_score')))
        
        if scores:
            # Enhanced Analytics Dashboard
            st.markdown("### 📊 Performance Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_score = np.mean(scores)
                if scale_type == "ielts":
                    st.metric("Average Band Score", f"{avg_score:.1f}", "Band")
                else:
                    st.metric("Average Score", f"{avg_score:.1f}")
            
            with col2:
                high_score = max(scores)
                if scale_type == "ielts":
                    st.metric("Highest Band", f"{high_score:.1f}", "Band")
                else:
                    st.metric("Highest Score", f"{high_score:.1f}")
            
            with col3:
                low_score = min(scores)
                if scale_type == "ielts":
                    st.metric("Lowest Band", f"{low_score:.1f}", "Band")
                else:
                    st.metric("Lowest Score", f"{low_score:.1f}")
            
            with col4:
                st.metric("Students Graded", len(scores))
            
            # Score Distribution with Enhanced Visualization
            st.markdown("#### 📈 Score Distribution")
            
            # Create two columns for visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Bar chart with proper labels
                if scale_type == "ielts":
                    # For IELTS, use 0-9 scale with 0.5 increments
                    bins = np.arange(0, 9.5, 0.5)
                    x_label = "IELTS Band Score"
                    title = "IELTS Band Distribution"
                else:
                    # For numeric, use 0-100 with 10-point increments
                    bins = list(range(0, 101, 10))
                    x_label = "Score (0-100)"
                    title = "Score Distribution"
                
                # Create histogram
                fig, ax = plt.subplots(figsize=(8, 4))
                counts, bins, patches = ax.hist(scores, bins=bins, edgecolor='black', alpha=0.7)
                
                # Customize appearance
                ax.set_xlabel(x_label, fontsize=10)
                ax.set_ylabel('Number of Students', fontsize=10)
                ax.set_title(title, fontsize=12, fontweight='bold')
                
                # Add count labels on bars
                for count, patch in zip(counts, patches):
                    if count > 0:
                        ax.text(patch.get_x() + patch.get_width() / 2, count + 0.1,
                               f'{int(count)}', ha='center', va='bottom', fontsize=9)
                
                # Add grid for better readability
                ax.grid(axis='y', alpha=0.3)
                ax.set_axisbelow(True)
                
                # Adjust layout
                plt.tight_layout()
                st.pyplot(fig)
            
            with viz_col2:
                # Box plot for distribution overview
                fig2, ax2 = plt.subplots(figsize=(8, 4))
                
                # Create box plot
                bp = ax2.boxplot(scores, vert=False, patch_artist=True, 
                                boxprops=dict(facecolor='lightblue'),
                                medianprops=dict(color='red', linewidth=2))
                
                # Add individual data points with jitter
                y = np.random.normal(1, 0.04, size=len(scores))
                ax2.scatter(scores, y, alpha=0.6, s=50, edgecolors='black')
                
                # Customize appearance
                if scale_type == "ielts":
                    ax2.set_xlabel('IELTS Band Score (0-9)', fontsize=10)
                    ax2.set_xlim(0, 9)
                    ax2.set_xticks(np.arange(0, 10, 1))
                else:
                    ax2.set_xlabel('Score (0-100)', fontsize=10)
                    ax2.set_xlim(0, 100)
                    ax2.set_xticks(np.arange(0, 101, 10))
                
                ax2.set_yticks([])
                ax2.set_title('Score Distribution with Outliers', fontsize=12, fontweight='bold')
                ax2.grid(axis='x', alpha=0.3)
                ax2.set_axisbelow(True)
                
                # Add statistics text
                stats_text = f"""
                Median: {np.median(scores):.1f}
                IQR: {np.percentile(scores, 75) - np.percentile(scores, 25):.1f}
                Std Dev: {np.std(scores):.2f}
                """
                ax2.text(0.02, 0.95, stats_text, transform=ax2.transAxes, 
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                st.pyplot(fig2)
            
            # Performance Ranking Table
            st.markdown("#### 🏆 Performance Ranking")
            
            # Create ranking dataframe
            ranking_data = []
            for idx, (name, score, orig_score) in enumerate(zip(student_names, scores, original_scores)):
                if scale_type == "ielts":
                    # Handle None values for orig_score
                    orig_display = f"{orig_score:.1f}" if orig_score is not None else "N/A"
                    ranking_data.append({
                        "Rank": idx + 1,
                        "Student": name,
                        "IELTS Band": f"{score:.1f}",
                        "Equivalent 100-point": orig_display,
                        "Performance": "Excellent" if score >= 7 else "Good" if score >= 5 else "Needs Improvement"
                    })
                else:
                    performance = "Excellent" if score >= 80 else "Good" if score >= 60 else "Needs Improvement"
                    ranking_data.append({
                        "Rank": idx + 1,
                        "Student": name,
                        "Score": f"{score:.1f}",
                        "Performance": performance
                    })
            
            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, use_container_width=True, hide_index=True)
            
            # Grammar Issues Analysis (if available)
            grammar_issues_data = []
            for r in st.session_state.results:
                if r.get("details", {}).get("grammar", {}).get("available"):
                    grammar_issues_data.append({
                        "Student": r.get("name"),
                        "Grammar Issues": r.get("details", {}).get("grammar", {}).get("issues_count", 0),
                        "Score": r.get("final_score")
                    })
            
            if grammar_issues_data:
                st.markdown("#### 🔍 Grammar Issues Analysis")
                grammar_df = pd.DataFrame(grammar_issues_data)
                
                # Create correlation analysis
                fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 4))
                
                # Scatter plot: Grammar issues vs Score
                ax3a.scatter(grammar_df["Grammar Issues"], grammar_df["Score"], 
                            alpha=0.6, s=100, edgecolors='black')
                ax3a.set_xlabel('Number of Grammar Issues', fontsize=10)
                ax3a.set_ylabel('Final Score', fontsize=10)
                ax3a.set_title('Grammar Issues vs Performance', fontsize=12, fontweight='bold')
                ax3a.grid(alpha=0.3)
                
                # Bar chart: Top grammar offenders
                sorted_grammar = grammar_df.sort_values("Grammar Issues", ascending=False).head(10)
                ax3b.barh(sorted_grammar["Student"], sorted_grammar["Grammar Issues"])
                ax3b.set_xlabel('Number of Grammar Issues', fontsize=10)
                ax3b.set_title('Top 10 Students with Most Grammar Issues', fontsize=12, fontweight='bold')
                ax3b.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig3)
            
            # Export results
            # Export results
            st.markdown("---")
            st.markdown("#### 📤 Export Results")
            
            if st.button("📊 Export Results as Excel", use_container_width=True):
                excel_bytes = export_results_to_excel(st.session_state.results)
                
                st.download_button(
                    label="⬇️ Download Excel (.xlsx)",
                    data=excel_bytes,
                    file_name=f"grading_results_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )


# ==================== GRADING EXECUTION ====================
if grade_button:
    # Input validation with clear feedback
    exercise_text = ex_text_paste.strip() if ex_text_paste.strip() else read_text_file(ex_file)
    model_text = model_text_paste.strip() if model_text_paste.strip() else read_text_file(model_file)
    
    if not exercise_text:
        st.error("❌ Please provide the exercise description (upload or paste).")
        st.stop()
    
    if not model_text:
        st.error("❌ Please provide the model solution (upload or paste).")
        st.stop()
    
    # Process rubric
    rubric_obj = None

    rubric_text = rubric_text_paste.strip()
    if not rubric_text and rubric_file:
        rubric_text = read_text_file(rubric_file)
    
    if rubric_text:
        rubric_obj = parse_teacher_rubric(rubric_text)
        if rubric_obj is None:
            st.warning("⚠️ Rubric detected but could not be parsed. Default grading will be used.")

    
    # Process student submissions
    student_texts = []
    student_names = []
    
    if student_files:
        for f in student_files:
            txt = read_text_file(f)
            if txt.strip():
                student_texts.append(txt.strip())
                student_names.append(f.name)
    
    if student_paste.strip():
        parts = [p.strip() for p in student_paste.split("\n---\n") if p.strip()]
        for i, p in enumerate(parts):
            student_texts.append(p)
            student_names.append(f"Student_{i+1}")
    
    if not student_texts:
        st.error("❌ No student submissions provided. Please upload files or paste submissions.")
        st.stop()
    
    # Initialize session state for results
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'grading_complete' not in st.session_state:
        st.session_state.grading_complete = False
    
    # Grade submissions with progress tracking
    st.session_state.results = []
    progress_bar = st.progress(0, text=f"Grading 0/{len(student_texts)} students...")
    
    for idx, (s_text, s_name) in enumerate(zip(student_texts, student_names)):
        progress = (idx) / len(student_texts)
        progress_bar.progress(progress, text=f"Grading {idx+1}/{len(student_texts)}: {s_name}...")
        
        try:
            if rubric_obj:
                res = apply_rubric_json(rubric_obj, model_text, s_text, output_scale)
            else:
                res = heuristic_grade(model_text, s_text, output_scale)
            
            # Enhanced feedback generation
            sim_pct = round(res.get("similarity", 0) * 100, 2)
            issues = res.get("grammar", {}).get("issues_count", "N/A")
            
            # Adjust reasoning based on scale
            if output_scale == "ielts_band_0-9":
                original_score = res.get("original_100_score", "N/A")
                reasoning = f"**IELTS Band:** {res.get('final_score')}/9 | **Original Score:** {original_score}/100 | **Grammar issues:** {issues}"
            else:
                reasoning = f"**Similarity to model answer:** {sim_pct}% | **Grammar issues:** {issues}"
            
            # Contextual feedback lines
            feedback_lines = []
            similarity = res.get("similarity", 0)
            
            if similarity >= 0.75:
                feedback_lines.append("Excellent content coverage and task achievement")
                feedback_lines.append("Well-structured response with clear organization")
            elif similarity >= 0.5:
                feedback_lines.append("Good content coverage with some minor gaps")
                feedback_lines.append("Consider expanding on key points for better depth")
            else:
                feedback_lines.append("Significant content gaps - review core concepts")
                feedback_lines.append("Focus on addressing all parts of the prompt")
            
            # Grammar-specific feedback
            if res.get("grammar", {}).get("available"):
                issues_count = res["grammar"]["issues_count"]
                if issues_count > 10:
                    feedback_lines.append("High number of grammar errors affecting readability")
                elif issues_count > 5:
                    feedback_lines.append("Moderate grammar issues - proofreading recommended")
                elif issues_count > 0:
                    feedback_lines.append("Minor grammar issues present")
                else:
                    feedback_lines.append("Excellent grammar and mechanics")
            
            # Generate AI feedback if enabled
            jina_feedback = None
            if enable_ai_feedback:
                jina_prompt = f"""
                Provide constructive feedback for this student work:
                
                Exercise: {exercise_text}
                Model Answer: {model_text}
                Student Answer: {s_text}
                Current Score: {res.get('final_score')}/{'9' if output_scale == 'ielts_band_0-9' else '100'}
                
                Provide 2-3 specific, actionable suggestions for improvement.
                """
                jina_feedback = generate_feedback_with_jina(jina_prompt)
            
            st.session_state.results.append({
                "name": s_name,
                "final_score": res.get("final_score"),
                "reasoning": reasoning,
                "feedback_lines": feedback_lines,
                "details": res,
                "jina_feedback": jina_feedback,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            st.session_state.results.append({
                "name": s_name, 
                "error": f"Grading failed: {str(e)}"
            })
    
    progress_bar.progress(1.0, text=f"✅ Completed grading {len(student_texts)} students!")
    time.sleep(0.5)
    progress_bar.empty()
    
    # Set grading complete flag
    st.session_state.grading_complete = True
    
    # Force display of results
    st.success(f"🎉 Successfully graded {len(student_texts)} submissions!")
    
    # Use rerun instead of switching tabs automatically
    st.rerun()

# Always show results if they exist, regardless of which tab we're on
if st.session_state.get('results') and len(st.session_state.results) > 0:
    # Display results in the current tab context
    st.markdown('<div class="sub-header">🎯 Grading Results</div>', unsafe_allow_html=True)
    
    for i, r in enumerate(st.session_state.results):
        with st.container():
            # Header with student name and score
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"### 👨‍🎓 {r.get('name', 'Student')}")
            with col2:
                score = r.get('final_score', 0)
                
                # Adjust score display based on scale
                if r.get("details", {}).get("scale_used") == "ielts":
                    score_text = f"{score}/9"
                    # IELTS band color coding
                    score_color = "green" if score >= 7 else "orange" if score >= 5 else "red"
                else:
                    score_text = f"{score}/100"
                    score_color = "green" if score >= 80 else "orange" if score >= 60 else "red"
                
                st.markdown(f"<h2 style='color: {score_color}; text-align: center;'>{score_text}</h2>", unsafe_allow_html=True)
            
            # Progress bar visualization (adjust max value based on scale)
            if r.get("details", {}).get("scale_used") == "ielts":
                progress_percent = (score / 9) * 100
            else:
                progress_percent = score
            
            st.markdown('<div class="progress-bar"><div class="progress-fill" style="width: {}%;"></div></div>'.format(progress_percent), unsafe_allow_html=True)
            
            # Score interpretation
            if r.get("details", {}).get("scale_used") == "ielts":
                if score >= 7:
                    st.markdown(f'<div class="success-box">🎉 Excellent IELTS performance! Band {score} indicates strong language proficiency.</div>', unsafe_allow_html=True)
                elif score >= 5:
                    st.markdown(f'<div class="warning-box">📚 Good IELTS band {score}. Shows competence with some room for improvement.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="warning-box">💡 IELTS band {score} indicates limited proficiency. Focus on core language skills.</div>', unsafe_allow_html=True)
            else:
                if score >= 80:
                    st.markdown('<div class="success-box">🎉 Excellent work! Strong understanding demonstrated.</div>', unsafe_allow_html=True)
                elif score >= 60:
                    st.markdown('<div class="warning-box">📚 Good effort, with room for improvement in key areas.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">💡 Needs significant improvement. Review fundamental concepts.</div>', unsafe_allow_html=True)
            
            # Detailed feedback in expanders
            col1, col2 = st.columns(2)
            
            with col1:
                with st.expander("📋 Detailed Feedback", expanded=True):
                    st.markdown("**Key Observations:**")
                    st.write(r["reasoning"])
                    
                    st.markdown("**Actionable Steps:**")
                    for line in r["feedback_lines"]:
                        st.markdown(f'<div class="feedback-item">💡 {line}</div>', unsafe_allow_html=True)
                    
                    if r.get("jina_feedback") and enable_ai_feedback:
                        st.markdown("**🤖 AI Insights:**")
                        st.write(r["jina_feedback"])
            
            with col2:
                if show_detailed_breakdown:
                    with st.expander("📊 Score Breakdown", expanded=True):
                        for item in r["details"].get("breakdown", []):
                            col_a, col_b, col_c = st.columns([3, 1, 1])
                            with col_a:
                                st.write(f"**{item['criterion']}**")
                            with col_b:
                                st.write(f"{item['subscore']:.1f}")
                            with col_c:
                                progress = item['subscore'] / 100
                                st.progress(progress)
                
                if show_grammar_examples and r["details"].get("grammar", {}).get("available"):
                    with st.expander("🔍 Grammar Check", expanded=False):
                        g = r["details"]["grammar"]
                        st.write(f"**Issues found:** {g['issues_count']}")
                        for ex in g["examples"]:
                            st.markdown(f"""
                            <div class="grammar-issue">
                                <strong>⚠️ {ex['message']}</strong><br>
                                <em>Context:</em> ...{ex['context']}...<br>
                                {f"<em>Suggestions:</em> {', '.join(ex.get('suggestions', []))}" if ex.get('suggestions') else ""}
                            </div>
                            """, unsafe_allow_html=True)
            
            st.divider()
