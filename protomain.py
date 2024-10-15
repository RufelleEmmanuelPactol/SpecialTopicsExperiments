import openai
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


from resume_scorer.free_form_text.kernels import NgramCrossProductSimilarity
from resume_scorer.resume_preprocessing.deconstructor import pdf_to_text
from io import BytesIO

from resume_scorer.keyword_processing.dynamic_algorithms import SimilarityScorer, format_resume, CrossProductSimilarity
import os
import dotenv
import numpy as np
import math
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

dotenv.load_dotenv()



def create_radar_chart(df):
    fig = go.Figure(data=go.Scatterpolar(
        r=df['Score'],
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


def resume_keyword_scorer_app():
    st.title("Resume Keyword Scorer")

    uploaded_file = st.file_uploader("Choose a PDF resume", type="pdf")

    keywords = ["Machine Learning", "Android Development", "Frontend Development", "Backend Development",
                "Data Analytics", "Development Operations", "Cyber Security", "Data Scraping", "Cloud Engineering",
                "Quantitative Analysis", "Data Engineering"]


    x = """st_tags(
        label='Keyword Introspection:',
        text='Add the keywords you want to use here...',
        suggestions=[],
        maxtags=-1,
    )"""

    if st.button("Start Introspection") and uploaded_file is not None and keywords:
        pdf_bytes = uploaded_file.read()
        resume_text = pdf_to_text(BytesIO(pdf_bytes))

        with st.spinner("Formatting Resume Post-Parse"):
            resume_text = format_resume(resume_text)

        st.subheader("Extracted Text from Resume")
        st.markdown(resume_text)

        st.subheader("Analysis Results")

        s1, s2 = st.columns(2)
        with s1:
            st.markdown("### Ngram-Product Engine")
            scorer = SimilarityScorer(engine='ngram-product', transformer='roberta-base-nli-stsb-mean-tokens')

            with st.spinner("Calculating Ngram-Product Engine Score"):
                relevance_scores = scorer.calculate_relevance_scores(resume_text, keywords)
            results_data = []
            for keyword, scores in relevance_scores.items():
                results_data.append({
                    'Keyword': keyword,
                    'Similarity Score': scores['similarity_score'],
                })

            results_df = pd.DataFrame(results_data)
            results_df = results_df.sort_values('Similarity Score', ascending=False)

            st.dataframe(results_df)

            st.subheader("Keyword Relevance Radar Chart")
            fig = create_radar_chart(
                results_df[['Keyword', 'Similarity Score']].rename(columns={'Similarity Score': 'Score'}))
            st.plotly_chart(fig)

            overall_score = results_df['Similarity Score'].mean()
            st.subheader(f"Overall Resume Score: {overall_score:.2f}")

        with s2:
            st.markdown("### Sentence-Chunk Engine")
            scorer = SimilarityScorer(engine='sentence-chunk', transformer='roberta-base-nli-stsb-mean-tokens')

            with st.spinner("Calculating Sentence-Chunk Engine Score"):
                relevance_scores = scorer.calculate_relevance_scores(resume_text, keywords)
            results_data = []
            for keyword, scores in relevance_scores.items():
                results_data.append({
                    'Keyword': keyword,
                    'Similarity Score': scores['similarity_score'],
                })

            results_df = pd.DataFrame(results_data)
            results_df = results_df.sort_values('Similarity Score', ascending=False)

            st.dataframe(results_df)

            st.subheader("Keyword Relevance Radar Chart")
            fig = create_radar_chart(
                results_df[['Keyword', 'Similarity Score']].rename(columns={'Similarity Score': 'Score'}))
            st.plotly_chart(fig)

            overall_score = results_df['Similarity Score'].mean()
            st.subheader(f"Overall Resume Score: {overall_score:.2f}")

    else:
        st.info("Please upload a PDF resume and enter keywords to analyze.")


def cross_product_similarity_app():
    st.title("Cross Product Similarity Scorer")

    text1 = st.text_area("Enter ANCHOR text:", height=200)
    text2 = st.text_area("Enter QUERY text:", height=200)

    scorer = CrossProductSimilarity(transformer='roberta-base-nli-stsb-mean-tokens')
    if len(text1) == 0 and len(text2) == 0:
        return
    if st.button("Calculate Similarity"):
        if text1 and text2:
            #c1 = st.columns(1)

            st.markdown("### Cross Product Score Generation Kernel")
            with st.spinner("Calculating correctness..."):
                similarity_score = scorer.calculate_similarity(text1, text2)

            st.subheader("Correctness Score")
            st.write(f"The correctness score is: {similarity_score:.4f}")

            st.header(f"{similarity_score*100:.2f}/100")

            if similarity_score > 0.8:
                st.success("The QUERY text is accurate")
            elif similarity_score > 0.5:
                st.info("The QUERY text is somewhat accurate.")
            else:
                st.warning("The QUERY text is quite incorrect.")
                z = """
            with c2:
                st.markdown("### Ngram Cross-Product Kernel")

                ngram_product = NgramCrossProductSimilarity(transformer='roberta-base-nli-stsb-mean-tokens')
                similarity = ngram_product.evaluate_similarity(text1, text2)
                st.write("The similarity between these two texts are the following: ", similarity)
                st.progress(similarity)
                """

        else:
            st.error("Please enter both texts to calculate similarity.")


def main():
    st.sidebar.title("App Selection")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Resume Keyword Scorer", "Cross Product Similarity", "Anchor Pooling"])

    if app_mode == "Resume Keyword Scorer":
        resume_keyword_scorer_app()
    elif app_mode == "Cross Product Similarity":
        cross_product_similarity_app()
    elif app_mode == "Anchor Pooling":
        anchor_pooling_app()


@st.cache_data
def generate_anchor_pool(question: str, num_answers: int = 7):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing diverse answers to questions."},
            {"role": "user",
             "content": f"Please provide {num_answers} unique and diverse answers to the following question. The answers are in paragraph. Separate each answer with '|||': {question}"}
        ],
        temperature=0.8,  # Increase variability in responses
        max_tokens=4000  # Adjust as needed
    )

    # Split the response into individual answers
    anchor_pool = response.choices[0].message.content.split('|||')

    # Clean up any leading/trailing whitespace
    anchor_pool = [answer.strip() for answer in anchor_pool]

    return anchor_pool


def anchor_pooling_app():
    st.title("Anchor Pooling Similarity Scorer")

    question = st.text_area("Enter question:", height=100)
    answer = st.text_area("Enter your answer:", height=200)

    scorer = CrossProductSimilarity(transformer='roberta-base-nli-stsb-mean-tokens')

    if st.button("Calculate Similarity"):
        if question and answer:
            with st.spinner("Generating anchor pool and calculating similarities..."):
                anchor_pool = generate_anchor_pool(question)

                similarities = []
                for idx, anchor in enumerate(anchor_pool, 1):
                    similarity = scorer.calculate_similarity(anchor, answer)
                    similarities.append({"Anchor": idx, "Similarity": similarity})

                df = pd.DataFrame(similarities)
                max_similarity = df['Similarity'].median()

                st.subheader("Similarity Scores")
                st.dataframe(df)

                st.subheader("Maximum Similarity Score")
                st.write(f"The maximum similarity score is: {max_similarity:.4f}")

                st.header(f"{max_similarity * 100:.2f}/100")

                if max_similarity > 0.8:
                    st.success("Your answer is highly similar to the best anchor.")
                elif max_similarity > 0.5:
                    st.info("Your answer has moderate similarity to the best anchor.")
                else:
                    st.warning("Your answer has low similarity to the best anchor.")

        else:
            st.error("Please enter both the question and your answer to calculate similarity.")


if __name__ == "__main__":
    main()