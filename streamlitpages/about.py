import streamlit as st
import pandas as pd

def write():
    st.header('Overview')
    st.markdown('This streamlit app uses a range of Covid-19 data sources that have been kindy curated by "Our World in Data" (OWID). \
    My intention with this app was was to design a user friendly interface for people to do their own exploratory analysis using\
    the Covid-19 OWID data.')

    st.subheader('Data')
    st.write('For a detailed breakdown of the data sources used by OWID, they have some great documentation on their\
     repo which can be found here: https://github.com/owid/covid-19-data/blob/master/public/data/README.md')

    st.markdown('To keep the app *light*, I did not include every single feature available in the data set but did \
    include the following:')

    # this is the data dictionary we show on the about page
    df_code = pd.read_csv('./public/data/owid-covid-codebook.csv')
    df_code = df_code[df_code.column.isin(cols_for_app)]
    st.dataframe(df_code)


    st.write("**Note** I can't be held accountable for the quality of the data in this dataset.")

    st.subheader('Libraries & Tools')
    st.write("This app was built in streamlit and all visualisations generated using Plotly Express or Plotly Graph Objects.\
    The code for the app is freely available on the app's repo on Github.")


    st.header('Github Links')
    st.write('OWID Repository : https://github.com/owid/covid-19-data')
    st.write('Forked Repository : https://github.com/kitsamho/covid-19-data')
    st.write('Streamlit .py file : https://github.com/kitsamho/covid-19-data')



    st.header('About Me | Contact Details')
    st.write('My name is Sam Ho. I am a Data scientist at Shutterstock AI')

    st.write('I would love to know what you thought of this app. If you have any comments or suggestions for improvement\
             (pretty sure there will be some bugs) please hit me up on LinkedIn: https://www.linkedin.com/in/kitsamho/')
    return