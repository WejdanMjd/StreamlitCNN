import streamlit as st
from Homepage import run_homepage  # Import the homepage function from Homepage.py

# Manage pages
def main():
    pages = {
        "Homepage": run_homepage,
    }
    page = st.sidebar.radio("Choose a page", list(pages.keys()))
    pages[page]()  # Run the selected page

if __name__ == "__main__":
    main()
