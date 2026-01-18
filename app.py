import streamlit as st
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

st.set_page_config(
    page_title="Job Fraud Detector",
    layout="wide"
)

st.title("Fake Job Posting Detector")
st.markdown("Enter the details of a job posting below to generate an Automatic Risk Assessment.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Metadata")
    title = st.text_input("Job Title", placeholder="e.g. Senior Software Engineer")
    department = st.text_input("Department", placeholder="e.g. Engineering")
    salary_range = st.text_input("Salary Range", placeholder="e.g. 100000-120000")
    
    st.write("Features")
    telecommuting = st.checkbox("Work from Home / Telecommuting?")
    has_company_logo = st.checkbox("Has Company Logo?")
    has_questions = st.checkbox("Has Screening Questions?")

with col2:
    st.subheader("Content")
    company_profile = st.text_area("Company Profile", height=100)
    description = st.text_area("Job Description", height=200)
    requirements = st.text_area("Requirements", height=100)
    benefits = st.text_area("Benefits", height=100)

st.divider()
predict_btn = st.button("Run Risk Analysis", type="primary", use_container_width=True)

if predict_btn:
    if not description:
        st.warning("Please enter at least a Job Description to analyze.")
    else:
        with st.spinner("Scanning for fraud patterns..."):
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

            st.write("---")
            st.subheader("Risk Analysis Report")

            if prediction[0] == 1:
                st.error("FLAG RAISED: This job posting has been identified as FRAUDULENT.")

            else:
                if (not has_company_logo) and (not salary_range):
                     st.warning("CAUTION: Content looks real, but critical verification is missing.")
                     st.write("System Flags:")
                     st.write("- **Identity Unknown:** No Company Logo.")
                     st.write("- **No Compensation:** Salary hidden.")
                     st.write("- **Analysis:** The text is professional, but the lack of metadata is highly suspicious for a remote role.")
                
                else:
                    st.success("PASSED: This job posting appears LEGITIMATE.")
                    st.write("Verification Checks:")
                    
                    if has_company_logo:
                        st.write("- Identity: Company logo present.")
                    else:
                        st.info("- Note: Missing company logo.")
                        
                    if len(description) > 1000:
                        st.write("- Detail: Description provides professional depth.")
                    
                    if has_questions:
                        st.write("- Screening: Employer is actively screening candidates.")