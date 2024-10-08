import streamlit as st
import pandas as pd
import plotly.graph_objects as go


from resume_scorer.free_form_text.kernels import CrossProductSimilarity, NgramCrossProductSimilarity
from resume_scorer.resume_preprocessing.deconstructor import pdf_to_text
from io import BytesIO

from resume_scorer.keyword_processing.dynamic_algorithms import SimilarityScorer, format_resume
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

    text1 = st.text_area("Enter first text:", height=200)
    text2 = st.text_area("Enter second text:", height=200)

    scorer = CrossProductSimilarity()
    if len(text1) == 0 and len(text2) == 0:
        return
    if st.button("Calculate Similarity"):
        if text1 and text2:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Cross Product Similarity Kernel")
                with st.spinner("Calculating similarity..."):
                    similarity_score = scorer.evaluate_similarity(text1, text2)

                st.subheader("Similarity Score")
                st.write(f"The similarity score between the two texts is: {similarity_score:.4f}")

                st.progress(similarity_score)

                if similarity_score > 0.8:
                    st.success("The texts are very similar!")
                elif similarity_score > 0.5:
                    st.info("The texts have moderate similarity.")
                else:
                    st.warning("The texts are quite different.")

            with c2:
                st.markdown("### Ngram Cross-Product Kernel")

                ngram_product = NgramCrossProductSimilarity(transformer='roberta-base-nli-stsb-mean-tokens')
                similarity = ngram_product.evaluate_similarity(text1, text2)
                st.write("The similarity between these two texts are the following: ", similarity)
                st.progress(similarity)



        else:
            st.error("Please enter both texts to calculate similarity.")


def main():
    st.sidebar.title("App Selection")
    app_mode = st.sidebar.selectbox("Choose the app mode",
                                    ["Resume Keyword Scorer", "Cross Product Similarity"])

    if app_mode == "Resume Keyword Scorer":
        resume_keyword_scorer_app()
    elif app_mode == "Cross Product Similarity":
        cross_product_similarity_app()


if __name__ == "__main__":
    main()