'''
Created on 

Course work: 

@author: Sanjjushri

Source:
    
'''

# Import necessary modules
# Global Import
import streamlit as st
import pandas as pd
from PIL import Image

#local import 
import toxic_comment_classifier as toxic

image = Image.open("beauty_prod.png")

def streamlit_app():

    st.set_page_config(layout = 'wide')

    st.title("Toxic Comment Predictor")

    st.text_area("Type Here")

    # html_temp = """
    # <div style = "background-color:blue; padding:10px">
    # <h5 style  = "color:white"; text-align:center; ">  </h5>
    # </div>
    # """

    # st.markdown(html_temp, unsafe_allow_html = True)

    st.image(image, caption='Comment your thoughts on this product')

    comment = st.text_input("Comment Here" )
    
    if st.button("Predict"):

        result = toxic.predict_comment(comment)

        if result[0] ==1:
            final = "The comment is under review will update once verified."

        else:
            final = "The comment has been successfully updated"

        st.header(final)
 


if __name__ == "__main__":
    streamlit_app()

