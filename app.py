import streamlit as st
from multimodal_search import MultimodalSearch

st.set_page_config(layout="wide")

def display_results(results):
    """Display the search results in columns."""
    for i, result in enumerate(results):
        col = st.columns(1)[0]
        with col:
            st.write(f"Score: {round(result.score * 100, 2)}%")
            st.image(result.content, use_column_width=True)

def main():
    st.markdown("<h1 style='text-align: center; color: yellow;'>Toy Search App</h1>", unsafe_allow_html=True)

    multimodal_search = MultimodalSearch()

    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if query:
            results = multimodal_search.search(query)
            st.warning(f"Your query was: {query}")
            st.subheader("Search Results:")
            
            if results:
                display_results(results)
            else:
                st.warning("No results found.")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()

