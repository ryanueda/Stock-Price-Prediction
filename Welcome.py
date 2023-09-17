import pandas as pd
import streamlit as st
import plotly.express as px
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
import base64
import io
import glob
import plotly.offline as pyo
import plotly.graph_objs as go
import streamlit.components.v1 as components




## HEADER
st.set_page_config(page_title='Landing Site', initial_sidebar_state='expanded')
st.markdown('<h1>Landing Site</h1>', unsafe_allow_html=True)
st.subheader("The Official Landing Site Of This Project")
st.write('')

st.write('\nWelcome :wave: This is the official landing site of my Personal Finances Analytics & Dashboard Web App.')


st.markdown('<h3>What Does This Website Do?</h3>', unsafe_allow_html=True)
st.markdown('''This website takes in DBS/POSB Bank Statements in the form of PDF, and uses <b>Optical Character Recognition (OCR)</b> technology 
         to scan the PDF for table data consisting of your transactions, and wrangles this data to produce a vast selection of interactive visualizations 
         and statistical analytics for you to better understand your own financial standing.''', unsafe_allow_html=True)
st.markdown('<div class="button"><a href="https://ryanueda.streamlit.app/Dashboard" target="_self"><p class="text">Go To Dashboard</p></a></div>', unsafe_allow_html=True)
st.write('')
st.write('')

st.markdown('<h3>How Do I Use It?</h3>', unsafe_allow_html=True)
st.markdown('''
            1) Log In to your e-Banking Portal
            2) Click Print to download transaction history according to the date range you desire in PDF, with the naming convention <b>'{bankname}\_statement\_{number}'</b>, with the oldest transactions being number 1, and newest being the largest number \n
            (<b>WARNING:</b> <i>Do Not Use Microsoft Print To PDF, use Save As PDF</i>) \n
            3) Head over to the <i>'Dashboard'</i> page
            4) Input your PDF Bank Statements in the sidebar File Upload widget & select your bank
            5) That's It! Allow our program some time to analyse your files, and view your personalized dashboard :)
            ''', unsafe_allow_html=True)
st.write('')
st.write('')
st.write('')

st.markdown('<h6>DISCLAIMERS</h6>', unsafe_allow_html=True)
st.markdown('''
            1) This app is in its beta testing stages, and may be unstable. If you encounter any bugs, feel free to drop me an email at <a href="ryanueda34@gmail.com">ryanueda34@gmail.com</a>
            2) Amount of time taken to analyse and generate dashboard scales proportionally with number of PDF files. This may take a while as the app is slightly computationally expensive and is hosted on Streamlit Sharing, with limited bandwidth
            ''', unsafe_allow_html=True)
st.write('')
st.write('')

st.markdown('<h6>DATA PRIVACY NOTICE</h6>', unsafe_allow_html=True)
st.markdown('''
            All data used in this site is strictly confidential and protected, and is not stored by any means. Additionally, no developers nor users are able to view your data other than yourself.
            All processes and workings of the site's functionaility is fully transparent and ethical.
            ''', unsafe_allow_html=True)


## SIDEBAR
st.sidebar.markdown('<h1 class="socials">My Socials</h1>', unsafe_allow_html=True)
st.sidebar.markdown('''<p class="socials">
               Interested in my works? Check out my <a href='https://github.com/ryanueda'>GitHub</a>, 
               or my <a href='https://linkedin.com/in/https://www.linkedin.com/in/ryan-ueda-teo-b32964212/'>LinkedIn</a>. 
               For my Personal Portfolio, you can view it <a href='https://ryanueda.netlify.app'>here</a>. </p>
                ''', unsafe_allow_html=True)

st.sidebar.markdown('\n')

st.sidebar.markdown('''<p class="socials">
               Liked this project? If you want to support my work, you can: &nbsp; &nbsp; 
                ''', unsafe_allow_html=True)

st.sidebar.markdown('''<p class="socials">
               <a href="https://www.buymeacoffee.com/ryanueda" target="_blank">
                <img class="coffee" src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" height="60">
               </a>
                ''', unsafe_allow_html=True)


st.sidebar.markdown('''
                    <p class="copyright">
                        <b>&#169; Copyright 2023 Ryan Ueda Teo</b> <br><br>
                        All Rights Reserved. <br>
                        Redistribution and use of source, with or without modification, are permitted provided it is approved by the <a href="ryanueda34@gmail.com">owner</a> and credit is given.
                    </p>
                    ''', unsafe_allow_html=True)


st.markdown("""
<style>

a {
    text-decoration: none;
    color: black;
}

.text {
color: black;
font-weight: bold;
margin-top: 15%;
}

.text:hover {
    text-decoration: none;
}

.button {
    width: 140px;
    height: 40px;
    background-color: #fff017;
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 25px;
    font-weight: bold;
    margin-left: -3px;
    color: black;
    transition: all 0.2s ease-in-out;
}

.button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
}

h1 {
    font-size: 55px;
    margin-bottom: 30px;
}

h3 {
    margin-top: 20px;
}

.coffee {
    border-radius: 17px;
    transition: all 0.2s ease-in-out;
}

.coffee:hover {
    transform: scale(1.05);
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
}

.socials {
    margin-left: 20px;  
}

.copyright {
    margin-top: 50%;
    font-size: 12px;
    margin-left: 7%;
}

</style>
""", unsafe_allow_html=True)
