import streamlit as st
import google.generativeai as genai
from langchain import PromptTemplate, LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI


#set up local environment
import os
from dotenv import load_dotenv
load_dotenv() #Activate the local environment
genai.configure(api_key = os.getenv('GOOGLE-API-KEY'))

#Designing the webpage
st.title("MARS(Automated Movie recommender system)✅")
movie_input = st.text_input('Enter the Movie title, genre or keyword 🎞️')

#prompt template
demo_template = '''Based on {movie_input}, give me best 5 movie  recommendations by their similarity of the synopsis of the film'''

template = PromptTemplate(input_variables = [movie_input], template = demo_template)

#initiate the model

llm = ChatGoogleGenerativeAI(model = 'gemini-pro', api_key = os.getenv('GOOGLE-API-KEY'))

#run the model
llm_chain = LLMChain(prompt = template, llm = llm)
llm_chain.run(movie_input)

if movie_input:
    prompt = template.format(movie_input = movie_input)
    recommendations = llm.predict(text = prompt)
    st.write(f'Recommendations for you:\n {recommendations}')
else:
    st.write('')