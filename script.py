import streamlit as st

def display_test_queue(queue):
    st.title("Ongoing Test Plan")

    for group_index, group in enumerate(queue, 1):
        with st.expander(f"Group {group_index}", expanded=True):
            cols = st.columns(len(group))
            for task, col in zip(group, cols):
                with col:
                    st.markdown(
                        f"""
                        <div style="
                            background-color: white;
                            border: 1px solid #ddd;
                            border-radius: 5px;
                            padding: 10px;
                            text-align: center;
                            height: 100%;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                        ">
                            <p style="margin: 0;">{task}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        st.markdown("<br>", unsafe_allow_html=True)

# Example usage
if __name__ == "__main__":
    example_queue = [["Mock Interviews Test", "Interview Modals Redesign V1"], ["Interview Modals Redesign V2", "Quiz Sign-Up Flow Reversal"]]
    display_test_queue(example_queue)