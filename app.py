from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from pytesseract import image_to_string
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
import streamlit as st
import multiprocessing
from tempfile import NamedTemporaryFile
import pandas as pd
import json
import requests
import base64

load_dotenv()

#1. convert PDF file into iamges via pypdfum2
def convert_pdf_to_images(file_path, scale=300/72):

    pdf_file = pdfium.PdfDocument(file_path)

    page_indices = [i for i in range(len(pdf_file))]

    renderer = pdf_file.render(
        pdfium.PdfBitmap.to_pil,
        page_indices=page_indices,
        scale=scale,
    )

    final_images = []

    for i, image in zip(page_indices, renderer):

        image_byte_array = BytesIO()
        image.save(image_byte_array, format='jpeg', optimize=True)
        image_byte_array = image_byte_array.getvalue()
        final_images.append(dict({i: image_byte_array}))

    return final_images


#2. extract text from images via pytesseract
def extract_text_from_img(list_dict_final_images):

    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for index, image_bytes in enumerate(image_list):

        image = Image.open(BytesIO(image_bytes))
        raw_text = str(image_to_string(image))
        image_content.append(raw_text)

    return "\n".join(image_content)

def extract_content_from_url(url: str):
    images_list = convert_pdf_to_images(url)
    text_with_pytesseract = extract_text_from_img(images_list)

    return text_with_pytesseract

#3. extract structured info from text via LLM
def extract_structured_data(content: str, data_points):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
    template = """
    You are an expert admin people who will extract core information from documents

    {content}

    Above is the content; please try to extract all data points from the content above 
    and export in a JSON array format:
    {data_points}

    Now please extract details from the content  and export in a JSON array format, 
    return ONLY the JSON array:
    """

    prompt = PromptTemplate(
        input_variables=["content", "data_points"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)

    results = chain.run(content=content, data_points=data_points)

    return results

#4. streamlit app

def main():
    st.title("PDF Content Extraction and Structuring")
    
    # 1. Upload the PDF file.
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
    if uploaded_file is not None:
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            file_path = temp_file.name
        
        # 2. Convert PDF file to images.
        st.write("Converting PDF to images...")
        images = convert_pdf_to_images(file_path)
        st.write(f"Converted {len(images)} pages.")

        # 3. Extract text from images.
        st.write("Extracting text from images...")
        extracted_text = extract_text_from_img(images)
        st.write("Text extraction completed.")
        
        # 4. Display a text box for inputting data points.
        data_points_input = st.text_area("Enter the data points you want to extract (comma-separated):")
        data_points = [x.strip() for x in data_points_input.split(",")]
        
        # 5. Extract structured information from the text.
        if st.button("Extract Structured Data"):
            st.write("Extracting structured data...")
            structured_data = extract_structured_data(extracted_text, data_points)
            st.write("Structured data extraction completed.")
            
            # 6. Display results and provide download link for JSON.
            st.json(structured_data)
            result_file = json.dumps(structured_data)
            b64 = base64.b64encode(result_file.encode()).decode()
            download_link = f'<a href="data:file/json;base64,{b64}" download="result.json">Download Result JSON</a>'
            st.markdown(download_link, unsafe_allow_html=True)

if __name__ == '__main__':
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method('spawn')
    main()