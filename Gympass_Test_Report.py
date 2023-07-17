import streamlit as st
from pathlib import Path


def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


def main():
    st.set_page_config(
        page_title="Gympass Test",
        page_icon="👋",
    )

    st.write("# Gympass Test 👋")
    st.sidebar.success("Click on the pages on the left side to check results.")
    readme_str = read_markdown_file("README.md")
    st.markdown(readme_str)


if __name__ == "__main__":
    main()
