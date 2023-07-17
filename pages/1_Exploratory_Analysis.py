import streamlit as st
from pathlib import Path
import re
import os
import base64

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()
def markdown_images(markdown):
    # example image markdown:
    # ![png](pages/assets/output_10_0.png)
    images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\))', markdown)
    return images



def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
    return markdown


def main():
    st.write("# Exploratory Analysis Notebook")
    readme_str = read_markdown_file("pages/assets/original_EDA.md")
    readme_str = markdown_insert_images(readme_str)
    st.markdown(readme_str,  unsafe_allow_html=True)
    
    


if __name__ == "__main__":
    main()
