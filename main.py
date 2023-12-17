import streamlit as st
import langch as lch
import textwrap


def main():
    """
    TubeGuidance - A Streamlit app for answering queries about YouTube videos based on their transcripts.

    Usage:
    - Enter the YouTube video URL and a query about the video.
    - The app retrieves information from the video transcript and provides detailed responses.

    """
    st.set_page_config(page_title='TubeGuidance')
    st.markdown('<h1 align="center">TubeGuidance</h1>', unsafe_allow_html=True)

    # User input form
    with st.form(key='my_form'):
        video_url = st.text_area(
            label="Could you provide the URL for the YouTube video?",
            max_chars=55
        )
        query = st.text_area(
            label="Feel free to inquire about the video",
            max_chars=45,
            key="query"
        )

        submit_button = st.form_submit_button(label='Submit')

    # Process user input and display response
    if query and video_url:
        # Create a database from the YouTube video URL
        db = lch.create_db_from_youtube_video_url(video_url)
        
        # Get response and associated documents from the query
        response, docs = lch.get_response_from_query(db, query)
        
        # Display response
        st.subheader("Response:")
        st.text(textwrap.fill(response, width=85))


if __name__ == '__main__':
    main()
