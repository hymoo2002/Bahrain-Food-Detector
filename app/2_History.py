import streamlit as st


st.set_page_config(
    page_title="History",
    page_icon="🕘",
)


st.title(" Prediction History")


history = st.session_state.get("history", [])


if not history:

    st.info("No predictions yet.")

else:

    for item in history:

        st.write(
            f"**{item['name']}**  \n"
            f"Confidence: {item['confidence']:.1%}  \n"
            f"{item['time']}"
        )

        st.divider()


    if st.button("Clear History"):

        st.session_state.history = []

        st.rerun()