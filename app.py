import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.pipeline.predict_pipeline import CustomData, PredictPipeline



# 1. Page Configuration
st.set_page_config(
    page_title="Job Fraud Detector",
    layout="wide"
)

# 2. Header
st.title("Fake Job Posting Detector")
st.markdown("Enter the details of a job posting below to check if it's **Real** or **Fraudulent**.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Job Metadata")
    
    title = st.text_input("Job Title", placeholder="e.g. Senior Software Engineer")
    
    department = st.text_input("Department", placeholder="e.g. Engineering")
    
    salary_range = st.text_input("Salary Range", placeholder="e.g. 100000-120000 (Leave empty if none)")
    
    telecommuting = st.checkbox("Work from Home / Telecommuting?")
    has_company_logo = st.checkbox("Has Company Logo?")
    has_questions = st.checkbox("Has Screening Questions?")

with col2:
    st.subheader("Job Content")
    
    company_profile = st.text_area("Company Profile", height=100, placeholder="Paste company description here...")
    description = st.text_area("Job Description", height=200, placeholder="Paste full job description here...")
    requirements = st.text_area("Requirements", height=150, placeholder="Paste requirements here...")
    benefits = st.text_area("Benefits", height=100, placeholder="Paste benefits here...")

st.divider()
predict_btn = st.button("Analyze Job Posting", type="primary", use_container_width=True)

if predict_btn:
    if not description:
        st.warning("Please enter at least a Job Description to analyze.")
    else:
        with st.spinner("Analyzing patterns against 17,000+ job records..."):
            data = CustomData(
                title=title,
                company_profile=company_profile,
                description=description,
                requirements=requirements,
                benefits=benefits,
                telecommuting=1 if telecommuting else 0,
                has_company_logo=1 if has_company_logo else 0,
                has_questions=1 if has_questions else 0,
                salary_range=salary_range,
                department=department
            )
            
            input_df = data.get_data_as_data_frame()
            
            pipeline = PredictPipeline()
            prediction = pipeline.predict(input_df)

            if prediction[0] == 1:
                st.error("FRAUD ALERT: This job posting appears to be FAKE.")
                st.write("Risk Factors Found:")
                if not has_company_logo:
                    st.write("- Missing Company Logo")
                if not salary_range:
                    st.write("- Missing Salary Information")
                if len(description) < 500:
                    st.write("- Description is unusually short")
            else:
                st.success("LEGITIMATE: This job posting appears to be REAL.")
                st.balloons()