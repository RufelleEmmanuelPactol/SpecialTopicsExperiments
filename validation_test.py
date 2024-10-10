import functools
import sys
from io import BytesIO
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
import pandas as pd
import streamlit as st
from blob_store import PersistentKVP
from resume_scorer.keyword_processing.dynamic_algorithms import SimilarityScorer, format_resume
import plotly.graph_objects as go
from resume_scorer.resume_preprocessing.deconstructor import pdf_to_text

def create_radar_chart(df):
    fig = go.Figure(data=go.Scatterpolar(
        r=df['Similarity Score'],
        theta=df['Keyword'],
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )

    return fig

@st.cache_resource
def get_kvp_store():
    return PersistentKVP()

kvp_store = get_kvp_store()

keywords = ["Machine Learning", "Android Development", "Frontend Development", "Backend Development",
                "Data Analytics", "Development Operations", "Cyber Security", "Data Scraping", "Cloud Engineering",
                "Quantitative Analysis", "Data Engineering"]

st.header("Resume - Keyword Fit Validation Test")
st.markdown(
    """
    This survey helps us understand the accuracy of our machine learning models and algorithms for skill-matching 
    through the exclusive use of resumes.
    """)

with st.expander("Our Privacy Policy", expanded=True):
    st.markdown("""
    ### Data Storage Disclaimer

    Your privacy and the security of your data are of utmost importance to us. By participating in this survey, you agree to the following terms regarding the collection and use of your data:

    - **Data Storage**: We only store the survey responses you provide. Please note that **your uploaded resume is not stored on our servers**. The resume is processed temporarily for analysis during this survey, but it is not saved or retained in any form after processing.

    - **Purpose of Data Collection**: The information collected through this survey will be used exclusively for academic research purposes. We aim to analyze the accuracy and effectiveness of our keyword-matching model as part of our study. Your data will not be used for any commercial purposes or shared with third parties outside of the research team.

    - **Confidentiality**: Your personal information, such as your name and ID number, will be kept confidential and will only be used to validate the survey responses. In any reports or publications resulting from this research, data will be presented in an aggregated manner, ensuring that individual responses cannot be identified.

    - **Voluntary Participation**: Participation in this survey is voluntary, and you may withdraw at any time. Should you choose to withdraw, any data collected up to that point will be deleted from our records.
    """)
    result = st.checkbox("By clicking agree, you acknowledge that you have read and understood this disclaimer, "
            "and you agree to the storage and use of your data as outlined above.")
if not result:
    exit(0)


st.markdown("#### Relevant Keywords")
st.write("We will be testing the following keywords and skills:")
st.text_area("Keywords", value=", ".join(keywords), height=100, disabled=True)

f1, f2 = st.columns(2)
with f1:
    first_name = st.text_input("First Name")
with f2:
    last_name = st.text_input("Last Name")
id_number = st.text_input("ID Number")

if id_number:
    if id_number in kvp_store:
        st.warning("This ID number has an associated submission. Please email rufelleemmanuel.pactol@cit.edu for removal.")
    else:
        st.success("ID number is valid.")

uploaded_resume = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])

if id_number and first_name and last_name and uploaded_resume is not None:
    st.markdown("### Thank you for providing your details and resume!")

    @st.cache_data
    def process_resume(file_content):
        resume_text = pdf_to_text(file_content)
        defaulted = format_resume(resume_text)
        with st.expander("Resume Parsed", expanded=False):
            st.markdown(defaulted)
        return defaulted

    @st.cache_data
    def calculate_scores(_resume_text, _keywords):
        scorer = SimilarityScorer(engine='ngram-product', transformer='roberta-base-nli-stsb-mean-tokens')
        return scorer.calculate_relevance_scores(_resume_text, _keywords)

    resume_text = process_resume(BytesIO(uploaded_resume.read()))
    relevance_scores = calculate_scores(resume_text, tuple(keywords))

    results_data = [{'Keyword': k, 'Similarity Score': v['similarity_score']} for k, v in relevance_scores.items()]
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Similarity Score', ascending=False)

    st.subheader("Model's Keyword Relevance Results")
    st.dataframe(results_df)

    st.markdown("""
    #### Guidelines to Interpreting The Data

| **Rating** | **Interpretation** |
|------------|--------------------|
| > 0.85     | The skills and qualifications are very prominent throughout the resume. This rating indicates that almost all of the mentioned skills and experiences closely match the specified keywords, showing a strong alignment with the required expertise. The candidate has demonstrated in-depth experience and mastery in these areas. |
| > 0.60     | The candidate is knowledgeable in the relevant skills and has demonstrated a good level of understanding and experience in the field. While not every aspect of the skills may be covered, there is clear evidence of familiarity and competence in these areas. |
| > 0.55     | The skills mentioned in the resume are somehow relevant to the specified keywords. There is a basic understanding or experience, but it may not be comprehensive. The candidate shows potential but might need further training or experience to achieve mastery. |
| > 0.50     | The candidate has skills that are related to the keywords, but the connection is more peripheral. The relevant skills may be mentioned in passing, or the experience may not directly align with the requirements. There is some foundational knowledge, but it might not be sufficient for in-depth work in these areas. |
| < 0.50     | The candidate is not proficient in the skills related to the specified keywords. The resume lacks significant mention or demonstration of these skills, indicating that the candidate might need substantial training or experience to reach the desired level of expertise. |

    """)

    st.subheader("Keyword Relevance Radar Chart")
    fig = create_radar_chart(results_df)
    st.plotly_chart(fig)

    st.subheader("Rate the Accuracy of Model's Predictions")
    st.markdown("Please rate the accuracy of the model's predictions for each keyword using the dropdown menus:")

    if 'ratings' not in st.session_state:
        st.session_state.ratings = {row['Keyword']: 3 for _, row in results_df.iterrows()}

    rating_options = {
        1: "1 - Not Accurate",
        2: "2 - Not Very Accurate",
        3: "3 - Somehow Accurate",
        4: "4 - Very Accurate",
        5: "5 - Extremely Accurate"
    }

    for i, row in results_df.iterrows():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write(row['Keyword'])
        with col2:
            st.selectbox(
                f"Rate accuracy for {row['Keyword']}",
                options=list(rating_options.keys()),
                format_func=lambda x: rating_options[x],
                key=f"rating_{row['Keyword']}",
                index=st.session_state.ratings[row['Keyword']] - 1,
                on_change=lambda: st.session_state.ratings.update({row['Keyword']: st.session_state[f"rating_{row['Keyword']}"]}),
            )

    st.header("Usability Testing Survey")
    st.subheader("Background Information")
    background_info = st.text_area("1. Can you tell me about your experience with online surveys, particularly those related to job applications or resume evaluations?")
    experience_similar = st.text_area("2. Have you used any similar keyword-matching tools or services before?")

    st.subheader("Expectations")
    expectations = st.text_area("3. What do you expect to achieve by participating in this survey?")
    specific_features = st.text_area("4. Are there specific features or aspects of the survey that you are particularly interested in?")

    st.subheader("Survey Navigation")
    navigation_ease = st.text_area("5. How easy was it to find and access the survey?")
    navigation_issues = st.text_area("6. Did you encounter any issues while navigating through the survey? If so, please describe them.")

    st.subheader("Understanding Instructions")
    instructions_clear = st.text_area("7. Were the instructions for uploading your resume clear and easy to follow?")
    purpose_understood = st.text_area("8. Did you understand the purpose of the survey and how your data would be used?")

    st.subheader("Resume Upload Process")
    upload_ease = st.text_area("9. How easy was it to upload your resume? Did you encounter any difficulties?")
    file_format_clear = st.text_area("10. Was the file format requirement (PDF only) clear to you?")

    st.subheader("Feedback on Keyword Relevance Results")
    results_clarity = st.text_area("11. How clear were the results provided by the model regarding keyword relevance?")
    rating_system_understood = st.text_area("12. Did you find the rating system (e.g., >0.85, >0.60) easy to understand? Why or why not?")

    st.subheader("Satisfaction Rating")
    satisfaction = st.slider("13. On a scale of 1 to 5, how satisfied are you with the overall survey experience?", 1, 5)
    satisfaction_feedback = st.text_area("14. What aspects of the survey did you find most helpful or frustrating?")

    st.header("General Usability Questions")
    overall_experience = st.text_area("15. How would you describe your overall experience with this survey?")
    likes = st.text_area("16. What did you like most about the survey? What did you like least?")

    st.header("Closing Questions")
    final_thoughts = st.text_area("19. Is there anything else you would like to add about your experience with the survey?")
    suggestions = st.text_area("20. Do you have any suggestions for improving the survey or the keyword-matching model?")
    future_participation = st.text_area("21. Would you be willing to participate in future surveys or usability tests related to this topic?")
    recommendation = st.text_area("22. Would you recommend this survey to others? Why or why not?")

    if st.button("Confirm and Submit"):
        #user_ratings = pd.DataFrame([(k, v) for k, v in st.session_state.ratings.items()], columns=['Keyword', 'Your Rating'])
        #comparison_df = results_df[['Keyword', 'Similarity Score']].merge(user_ratings, on='Keyword')
        #st.dataframe(comparison_df[['Keyword', 'Similarity Score', 'Your Rating']])

        normalized_results = {}
        for keyword in keywords:
            similarity_score = results_df[results_df['Keyword'] == keyword]['Similarity Score'].values[0]
            user_rating = st.session_state.ratings[keyword]

            normalized_results[keyword] = {
                'similarity_score': float(similarity_score),
                'user_rating': int(user_rating)
            }

        survey_data = {
            "background_info": background_info,
            "experience_similar": experience_similar,
            "expectations": expectations,
            "specific_features": specific_features,
            "navigation_ease": navigation_ease,
            "navigation_issues": navigation_issues,
            "instructions_clear": instructions_clear,
            "purpose_understood": purpose_understood,
            "upload_ease": upload_ease,
            "file_format_clear": file_format_clear,
            "results_clarity": results_clarity,
            "rating_system_understood": rating_system_understood,
            "satisfaction": satisfaction,
            "satisfaction_feedback": satisfaction_feedback,
            "overall_experience": overall_experience,
            "likes": likes,
            "final_thoughts": final_thoughts,
            "suggestions": suggestions,
            "future_participation": future_participation,
            "recommendation": recommendation
        }

        results = {
            'first_name': first_name,
            'last_name': last_name,
            'id_number': id_number,
            'keywords': normalized_results,
            'survey_data': survey_data
        }
        kvp_store[id_number] = results

        st.success("Results saved successfully!")
        st.cache_data.clear()