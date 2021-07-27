import streamlit as st
st.set_page_config(layout="wide")
from PIL import Image
import os
from streamlitpages import about, cross_section_analysis, time_series_analysis
from helperfunctions.helper import *
pd.set_option('display.float_format', lambda x: '%.0f' % x)

# these are the OWID columns we want to transform using either fillna or interpolation
transform_cols = ['people_vaccinated_per_hundred',
                  'total_vaccinations',
                  'total_deaths',
                  'total_deaths_per_million',
                  'total_cases_per_million',
                  'icu_patients_per_million',
                  'hosp_patients_per_million']

cols_for_app = ['continent', 'location', 'date', 'total_deaths', 'total_deaths_per_million', \
                'total_cases_per_million', 'icu_patients_per_million', 'hosp_patients_per_million',
                'people_vaccinated_per_hundred', 'total_vaccinations',
                'gdp_per_capita', 'population', 'stringency_index', 'population',
                'population_density', 'median_age', 'aged_65_older',
                'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
                'cardiovasc_death_rate', 'diabetes_prevalence',
                'female_smokers', 'male_smokers', 'handwashing_facilities',
                'hospital_beds_per_thousand', 'life_expectancy']

owid_csv_path = os.getcwd() + '/public/data/owid-covid-data.csv'
owid_code_book_path = os.getcwd()+'/public/data/owid-covid-codebook.csv'


@st.cache
def get_owid_data(csv_path, cols):
    df = pd.read_csv(csv_path)
    df = df[cols]
    return df

def get_owid_codebook(csv_path, cols):
    df_code = pd.read_csv(csv_path)
    df_code = df_code[df_code.column.isin(cols)]
    df_code = df_code.rename(columns={'column':'variable'})
    return df_code

# get original OWID DataFrame
df = get_owid_data(owid_csv_path, cols_for_app)

# get code book
df_code_book = get_owid_codebook(owid_code_book_path, cols_for_app)


@st.cache
def format_owid_data(df, transform_cols):
    """ This is the main function that transforms the raw OWID data into something we can use in the app
    Args:
        Original DataFrame from csv
    Returns:
        Processed / cleaned DataFrame
    """

    # loop through and subset each country to a list
    country_dfs = []

    # loop through each country
    for country in df.location.unique():
        df_country = df[df.location == country]  # df masked on country
        df_country.date = pd.to_datetime(df_country.date)  # convert string date to datetime
        df_country = df_country.sort_values(by='date')  # sort by date

        # transform our continuous columns
        for col in transform_cols:
            df_country[col] = update_series(df_country[col]).astype(int)

        # we will group by week and use max as agg so each row will represent the max value in any given week
        df_country = df_country.groupby(pd.Grouper(key='date', freq='W')).max()

        country_dfs.append(df_country)  # append unique country dataframe to list

    df_final = pd.concat(country_dfs)

    df_final = df_final.reset_index().sort_values(by=['location', 'date'])

    # if there any remaining nulls we can replace them
    df_final = df_final.fillna(0)

    # select start point and sort date in ascending order
    df_final = df_final.sort_values(by='date', ascending=True)
    df_final = df_final[df_final.date >= '2020-02-09']

    # date needs to be in string format for plotly animations to work
    df_final.date = df_final.date.astype(str)
    df_final = df_final[df_final.continent != 0]

    # get rid of any duplicate columns
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    return df_final


# format DataFrame
df_final = format_owid_data(df, transform_cols)

# this initialises the app and sets up the template with tabs
def streamlit_init():

    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True)
    query_params = st.experimental_get_query_params()
    tabs = ["About", "Covid-19 Cross Section", 'Covid-19 Time Series']

    im = Image.open('./covid_streamlit_app_assets/logo.jpeg')
    st.image(im.resize((int(im.size[0] / 1), int(im.size[1] / 1)), 0))

    if "tab" in query_params:
        active_tab = query_params["tab"][0]
    else:
        active_tab = "About"

    if active_tab not in tabs:
        st.experimental_set_query_params(tab="About")
        active_tab = "Home"

    li_items = "".join(
        f"""
        <li class="nav-item">
            <a class="nav-link{' active' if t == active_tab else ''}" href="/?tab={t}">{t}</a>
        </li>
        """
        for t in tabs
    )
    tabs_html = f"""
        <ul class="nav nav-tabs">
        {li_items}
        </ul>
    """

    st.markdown(tabs_html, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    return active_tab


active_tab = streamlit_init()

if active_tab == "About":
    about.write(df_code_book)

elif active_tab == "Covid-19 Cross Section":
    cross_section_analysis.write(df_final)

elif active_tab == "Covid-19 Time Series":
    time_series_analysis.write(df_final)







