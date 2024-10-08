import functools
import sys
from io import BytesIO
import dotenv
dotenv.load_dotenv()

import pandas as pd
import streamlit as st
from blob_store import PersistentKVP
from resume_scorer.keyword_processing.dynamic_algorithms import SimilarityScorer, format_resume
import plotly.graph_objects as go
from resume_scorer.resume_preprocessing.deconstructor import pdf_to_text
user_rankings = None
def create_interactive_table(df):
    if 'rankings' not in st.session_state:
        st.session_state.rankings = {row['Keyword']: len(df) for _, row in df.iterrows()}

    for i, row in df.iterrows():
        col1, col4 = st.columns([3, 2])
        with col1:
            st.write(row['Keyword'])
        with col4:
            st.number_input(f"Rank for {row['Keyword']}",
                            min_value=1,
                            max_value=len(df),
                            value=st.session_state.rankings[row['Keyword']],
                            key=f"rank_{row['Keyword']}",
                            on_change=update_ranking,
                            args=(row['Keyword'],))

    df_result =  pd.DataFrame([(k, v) for k, v in st.session_state.rankings.items()], columns=['Keyword', 'Your Rank'])
    if len(df_result['Your Rank'].unique()) < len(df_result['Your Rank']):
        st.warning("Please make sure that your provided rankings are unique.")
        return
    resulters = df_result['Your Rank']

    return df_result

def update_ranking(keyword):
    st.session_state.rankings[keyword] = st.session_state[f"rank_{keyword}"]

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
    return PersistentKVP("local-db")

kvp_store = get_kvp_store()

keywords = ["Machine Learning", "Application Development", "Backend Development", "Frontend Development",
            "Data Analytics", "Water Plumbing", "Medical Research"]

st.header("Resume - Keyword Fit Validation Test")
st.markdown(
    """
    For this survey, we are testing the accuracy of our keyword-matching model and algorithm.
    The purpose of this algorithm is to assess how well certain skills, qualifications, and keywords 
    align with the content of your resume. This helps us determine if the model accurately identifies 
    the relevant skills for job applications based on the keywords.
    """
)

st.markdown("### Keywords Analyzed")
st.write("These are the keywords that the algorithm has identified as relevant:")
st.text_area("Keywords", value=", ".join(keywords), height=100, disabled=True)

first_name = st.text_input("First Name")
last_name = st.text_input("Last Name")
id_number = st.text_input("ID Number")

if id_number:
    if id_number in kvp_store:
        st.warning("This ID number is not valid. Please check your ID and try again.")
    else:
        st.success("ID number is valid.")

uploaded_resume = st.file_uploader("Upload your resume (PDF format only)", type=["pdf"])

if id_number and first_name and last_name and uploaded_resume is not None:
    st.markdown("### Thank you for providing your details and resume!")
    st.markdown("Please proceed to rank the keywords based on their relevance to your resume.")

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
    results_df['Model Rank'] = results_df['Similarity Score'].rank(ascending=False, method='min')

    st.subheader("Keyword Relevance and Ranking")
    st.write("Please rank the keywords based on their relevance to your resume (1 being most relevant):")
    user_rankings = create_interactive_table(results_df)
    if user_rankings is None:
        sys.exit(0)



if st.button("Confirm and Submit"):
    comparison_df = results_df[['Keyword', 'Model Rank']].merge(user_rankings, on='Keyword')
    st.dataframe(comparison_df[['Keyword', 'Model Rank', 'Your Rank']])

    st.subheader("Keyword Relevance Radar Chart")
    fig = create_radar_chart(results_df[['Keyword', 'Similarity Score']])
    st.plotly_chart(fig)


# Move this block outside of the "Confirm" button condition
if user_rankings is not None:
    if True:
        normalized_results = {}
        for keyword in keywords:
            model_rank = results_df[results_df['Keyword'] == keyword]['Model Rank'].values[0]
            user_rank = user_rankings[user_rankings['Keyword'] == keyword]['Your Rank'].values[0]
            similarity_score = results_df[results_df['Keyword'] == keyword]['Similarity Score'].values[0]

            normalized_results[keyword] = {
                'model_rank': int(model_rank),
                'user_rank': int(user_rank),
                'similarity_score': float(similarity_score)
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