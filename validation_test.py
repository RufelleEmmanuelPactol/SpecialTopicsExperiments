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
user_ratings = None

def create_interactive_table(df):
    if 'ratings' not in st.session_state:
        st.session_state.ratings = {row['Keyword']: 3 for _, row in df.iterrows()}

    rating_options = {
        1: "1 - Not Accurate",
        2: "2 - Slightly Accurate",
        3: "3 - Moderately Accurate",
        4: "4 - Very Accurate",
        5: "5 - Extremely Accurate"
    }

    for i, row in df.iterrows():
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write(row['Keyword'])
        with col2:
            selected_rating = st.selectbox(
                f"Rate accuracy for {row['Keyword']}",
                options=list(rating_options.keys()),
                format_func=lambda x: rating_options[x],
                key=f"rating_{row['Keyword']}",
                index=st.session_state.ratings[row['Keyword']] - 1
            )
            st.session_state.ratings[row['Keyword']] = selected_rating

    df_result = pd.DataFrame([(k, v) for k, v in st.session_state.ratings.items()], columns=['Keyword', 'Your Rating'])
    return df_result

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

keywords = ["Machine Learning", "Application Development", "Backend Development", "Frontend Development",
            "Data Analytics", "Water Plumbing", "Medical Research"]

st.header("Resume - Keyword Fit Validation Test")
st.markdown(
    """
    This survey helps us understand the accuracy of our machine learning models and algorithms for skill-matching 
    through the exclusive use of resumes.
    """)

with st.expander("Our Privacy Policy", expanded=False):
    st.markdown("""
    ### Data Storage Disclaimer

    Your privacy and the security of your data are of utmost importance to us. By participating in this survey, you agree to the following terms regarding the collection and use of your data:

    - **Data Storage**: We only store the survey responses you provide. Please note that **your uploaded resume is not stored on our servers**. The resume is processed temporarily for analysis during this survey, but it is not saved or retained in any form after processing.

    - **Purpose of Data Collection**: The information collected through this survey will be used exclusively for academic research purposes. We aim to analyze the accuracy and effectiveness of our keyword-matching model as part of our study. Your data will not be used for any commercial purposes or shared with third parties outside of the research team.

    - **Confidentiality**: Your personal information, such as your name and ID number, will be kept confidential and will only be used to validate the survey responses. In any reports or publications resulting from this research, data will be presented in an aggregated manner, ensuring that individual responses cannot be identified.

    - **Voluntary Participation**: Participation in this survey is voluntary, and you may withdraw at any time. Should you choose to withdraw, any data collected up to that point will be deleted from our records.
    """)
st.markdown("By proceeding with this survey, you acknowledge that you have read and understood this disclaimer, "
            "and you agree to the storage and use of your data as outlined above.")

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
    st.markdown("Please rate the accuracy of the model's predictions for each keyword using the dropdown menus.")

    @st.cache_data
    def process_resume(file_content):
        resume_text = pdf_to_text(file_content)
        return format_resume(resume_text)

    @st.cache_data
    def calculate_scores(_resume_text, _keywords):
        scorer = SimilarityScorer(engine='ngram-product', transformer='roberta-base-nli-stsb-mean-tokens')
        return scorer.calculate_relevance_scores(_resume_text, _keywords)

    resume_text = process_resume(BytesIO(uploaded_resume.read()))
    relevance_scores = calculate_scores(resume_text, tuple(keywords))

    results_data = [{'Keyword': k, 'Similarity Score': v['similarity_score']} for k, v in relevance_scores.items()]
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Similarity Score', ascending=False)

    st.subheader("Keyword Relevance and Rating")
    st.info("Please rate the accuracy of the model's predictions for each keyword using the dropdown menus:")
    user_ratings = create_interactive_table(results_df)

if st.button("Confirm and Submit"):
    comparison_df = results_df[['Keyword', 'Similarity Score']].merge(user_ratings, on='Keyword')
    st.dataframe(comparison_df[['Keyword', 'Similarity Score', 'Your Rating']])

    st.subheader("Keyword Relevance Radar Chart")
    fig = create_radar_chart(results_df[['Keyword', 'Similarity Score']])
    st.plotly_chart(fig)

if user_ratings is not None:
    if True:
        normalized_results = {}
        for keyword in keywords:
            similarity_score = results_df[results_df['Keyword'] == keyword]['Similarity Score'].values[0]
            user_rating = user_ratings[user_ratings['Keyword'] == keyword]['Your Rating'].values[0]

            normalized_results[keyword] = {
                'similarity_score': float(similarity_score),
                'user_rating': int(user_rating)
            }

        results = {
            'first_name': first_name,
            'last_name': last_name,
            'id_number': id_number,
            'keywords': normalized_results
        }
        kvp_store[id_number] = results

        st.success("Results saved successfully!")
        st.cache_data.clear()