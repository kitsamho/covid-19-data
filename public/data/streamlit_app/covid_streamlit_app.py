import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import normalize
from PIL import Image
pd.set_option('display.float_format', lambda x: '%.0f' % x)

# this is a plotly figure formatting function that can be used on a range of Plotly figure objects
def plotly_streamlit_layout(fig, barmode=None, barnorm=None, height=None,width=None):
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      barmode=barmode,
                      barnorm=barnorm,
                      height = height,
                      width = width)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black')

    fig.update_layout(margin=dict(l=50, r=50, b=50, t=50, pad=2))
    fig.update_layout(bargap=0.03)

    return fig

# this is a plotly figure formatting function (text) that can be used on a range of Plotly figure objects
def plotly_streamlit_texts(fig, x_title, y_title):
    fig.update_layout(yaxis=dict(title=y_title, titlefont_size=10, tickfont_size=10),
                      xaxis=dict(title=x_title, titlefont_size=10, tickfont_size=10))

    return fig

# this generates a heatmap from a Pandas.corr() DataFrame
def get_heatmap(df):
    mask = np.triu(np.ones_like(df, dtype=bool))
    data = df.mask(mask)

    heat = go.Heatmap(z=data,
                      x=data.columns.values,
                      y=data.columns.values,
                      zmin=- 0.01,
                      zmax=1,
                      xgap=1,
                      ygap=1,
                      colorscale='Greens')

    layout = go.Layout(
        title_x=0.5,
        width=800,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        yaxis_autorange='reversed'
    )

    fig = go.Figure(data=[heat], layout=layout)
    return fig

# this function returns index ranges where there are nulls
def get_indexes(x):
    index_fill_1 = [i for i in range(x.index[0], x.dropna().index[0])]
    index_interpolate = [i for i in range(x.dropna().index[0], x.index[-1])]
    return index_fill_1, index_interpolate

# this function updates a series of data using either fill na or interpolation
def update_series(x):
    if len(x.dropna()) == 0:
        x = x.fillna(1)
        return x
    else:
        index_fill_1, index_interpolate = get_indexes(x)
        x_fill_1 = x[x.index.isin(index_fill_1)]
        x_interpolate = x[x.index.isin(index_interpolate)]
        x_fill_1 = x_fill_1.fillna(1)
        x_interpolate = x_interpolate.interpolate()
        return pd.concat([x_fill_1, x_interpolate])

# these are the OWID columns we want to transform using either fillna or interpolation
transform_cols = ['people_vaccinated_per_hundred',
                  'total_vaccinations',
                  'total_deaths',
                  'total_deaths_per_million',
                  'total_cases_per_million',
                  'icu_patients_per_million',
                  'hosp_patients_per_million']


@st.cache
def get_data(df, transform_cols):
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


# read in the raw OWID data
df = pd.read_csv('../owid-covid-data.csv')

# select the columns we want
cols_for_app = ['continent','location','date','total_deaths','total_deaths_per_million',\
                'total_cases_per_million','icu_patients_per_million','hosp_patients_per_million',
                'people_vaccinated_per_hundred','total_vaccinations',
                'gdp_per_capita','population','stringency_index','population',
                'population_density', 'median_age', 'aged_65_older',
                'aged_70_older', 'gdp_per_capita', 'extreme_poverty',
                'cardiovasc_death_rate', 'diabetes_prevalence',
                'female_smokers','male_smokers', 'handwashing_facilities',
                'hospital_beds_per_thousand','life_expectancy']

df = df[cols_for_app]

# get the data
df_final = get_data(df, transform_cols)

# drop any nulls
df.dropna(subset=['population'], inplace=True)


# this initialises the app and sets up the template with tabs
def streamlit_init():

    st.markdown(
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
        unsafe_allow_html=True)
    query_params = st.experimental_get_query_params()
    tabs = ["About", "Covid-19 Cross Section Analysis", 'Covid-19 Time Series Analysis']

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
    df_code = pd.read_csv('../owid-covid-codebook.csv')
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

elif active_tab == "Covid-19 Cross Section Analysis":

    a = st.beta_expander('About (click to expand)')

    a.write("The crossplot analysis allows you to plot two features against one another with the marker sizes representing \
    what we might consider as dependant features e.g. total deaths, total deaths per million, total vaccinations.")
    a.write("DataFrame masking allows you to explore the data by continent or include all countries.")
    a.write("Each feature's central tendency is represented by the dashed line on each axis so you can see where \
    countries are positioned in terms of the distribution for each feature. These update when you change features.")
    a.write('The heat map is a summary of the correlations between all features')

    # DataFrame set up
    df_country_unique = df_final[df_final.columns[1:3]].drop_duplicates(subset=['location'], keep='first').set_index(
        'location')
    df_analysis = pd.concat([df_country_unique, df_final.groupby('location')[df_final.columns[3:]].max()], axis=1)
    df_analysis = df_analysis.reset_index()
    df_analysis.rename(columns={'index': 'location'}, inplace=True)

    st.subheader('Crossplots')

    # a dictionary of string representation keys and feature values (independent variables)
    variable_dic = {'GDP per Capita': 'gdp_per_capita',
                    'Population': 'population',
                    'Stringency Index': 'stringency_index',
                    'Population Density': 'population_density',
                    'Median Age': 'median_age',
                    'Aged 65 or older': 'aged_65_older',
                    'Aged 70 or older': 'aged_70_older',
                    'Extreme Poverty': 'extreme_poverty',
                    'Cardiovascular Death Rate': 'cardiovasc_death_rate',
                    'Diabetes Prevalance':'diabetes_prevalence',
                    'Female Smokers':'female_smokers',
                    'Male Smokers': 'male_smokers',
                    'Handwashing Facilities': 'handwashing_facilities',
                    'Hospital Beds per Thousand':'hospital_beds_per_thousand',
                    'Hospital patients per million':'hosp_patients_per_million',
                    'Life Expectancy': 'life_expectancy'}

    # a dictionary of string representation keys and feature values (dependent variables)
    size_dic = {'Total Deaths': 'total_deaths',
                'Total Deaths per Million': 'total_deaths_per_million',
                'Total Cases per Million': 'total_cases_per_million',
                'People Vaccinated per Hundred': 'people_vaccinated_per_hundred',
                'No sizing': 'no_sizing'}

    # set up some columns for the interactive widgets - use a mid point to create some buffer between widgets
    c1, mid, c2, = st.beta_columns((3,0.5,3))

    # set up the widget options for continents
    continents = list(df_analysis.continent.unique())
    continents.sort()
    continents.insert(0,'Show All Countries')

    # set up widget options for x,y, scatter marks and get values from dic
    x_metric = c1.selectbox('X axis', list(variable_dic.keys()), 1)
    x = variable_dic[x_metric]

    y_metric = c1.selectbox('Y axis', list(variable_dic.keys()), 4)
    y = variable_dic[y_metric]

    size_by = c2.selectbox('Size markers by', list(size_dic.keys()), 0)
    size_by = size_dic[size_by]

    # if we want to get relative sizes of markers set up plot frame including variable to size
    if size_by != 'no_sizing':
        df_plot = df_analysis[[x, y, size_by, 'location', 'continent']]
        rel_sizing = True
    else:
        df_plot = df_analysis[[x, y, 'location', 'continent']]
        rel_sizing = False

    # select continent
    df_continent = c2.selectbox('Show which continent', continents, 0)

    # marker size adjustment
    if size_by != 'no_sizing':
        marker_size = st.slider('Use this to adjust relative marker size', 1, 1000, step=1,value=150)
    else:
        marker_size = st.slider('Use this to adjust marker size', 1, 30, step=1, value=10)

    # select which measure of central tendency to show
    average_kind = c1.selectbox('Which central tendency', ('Mean', 'Median'), 0)

    # if people have selected a specific continent to show then mask DataFrame
    if df_continent != 'Show All Countries':
        df_plot = df_plot[df_plot.continent == df_continent]

    # function that adjusts the relative marker size using sklearn.preprocessing.normalize
    def reshape_for_plot(df, col, marker_size):
        return pd.Series((normalize([df[col]]) * marker_size)[0])
    
    # main function that generates plot
    def plot_scatter(df, x, y, size, marker_size, rel_sizing =True, average_kind ='Mean'):

        fig_scatter = go.Figure() # get a graph objects figure
        df_to_plot = df.reset_index(drop=True)
        col = px.colors.qualitative.Plotly * 25 # get 25 colours

        # if we want relative sizing
        if rel_sizing:
            plot_size = reshape_for_plot(df_to_plot, size, marker_size)
            hover_texts = df_to_plot[size]
        else:
            plot_size = [marker_size] * df_to_plot.shape[0]
            hover_texts = [''] * df_to_plot.shape[0]

        # loop through each row and add marker and various bits of meta data
        for i in range(df_to_plot.shape[0]):
            fig_scatter.add_trace(go.Scatter(
                x=np.array(df_to_plot[x][i]),
                y=np.array(df_to_plot[y][i]),
                name=df_to_plot['location'][i],
                hovertext='<b>' + df_to_plot['location'][i] + '</b>' + '<br>' + size.capitalize().replace('_', ' ') + ' : ' + \
                          str(hover_texts[i]),
                hoverinfo="text",
                mode='markers',

                # marker size is adjusted using a reshape function
                marker=dict(size=plot_size[i], opacity=0.5,
                            color=col[i])))

        # add vertical and horizontal lines to represent mean or median
        if average_kind == 'Mean':
            fig_scatter.add_vline(x=df_to_plot[x].mean(), line_width=1, line_dash="dash", line_color="grey")
            fig_scatter.add_hline(y=df_to_plot[y].mean(), line_width=1, line_dash="dash", line_color="grey")
        else:
            fig_scatter.add_vline(x=df_to_plot[x].median(), line_width=1, line_dash="dash", line_color="grey")
            fig_scatter.add_hline(y=df_to_plot[y].median(), line_width=1, line_dash="dash", line_color="grey")

        # use log scale for x and y - makes plot more readable
        fig_scatter.update_xaxes(type="log")
        fig_scatter.update_yaxes(type="log")

        # some formatting of plot - background colours and show legend
        fig_scatter.update_layout(legend={'itemsizing': 'constant'})
        fig_scatter.update_layout(plot_bgcolor='white',
                                  width=1000,
                                  height=1000,
                                  showlegend=True)

        # some formatting of plot - axis font size and tick size
        fig_scatter.update_layout(yaxis=dict(title=y.capitalize().replace('_', ' '), titlefont_size=15, tickfont_size=10),
                                  xaxis=dict(title=x.capitalize().replace('_',' '), titlefont_size=15, tickfont_size=10))

        return fig_scatter


    # get plot
    fig_cross_plot = plot_scatter(df_plot, x=x, y=y, size=size_by, rel_sizing=rel_sizing,marker_size=marker_size)

    # add plot to streamlit
    st.plotly_chart(plotly_streamlit_layout(fig_cross_plot, height=800, width=1600))

    # Show Spearman's correlation co-efficient of variables chosen
    df_corr = df_plot[[x,y]].corr()
    corr_val = str("{:.2f}".format(df_corr[df_corr.columns[1]].iloc[0]))

    st.markdown(f"Correlation of **{x.capitalize().replace('_',' ')}** and **{y.capitalize().replace('_',' ')}** = \
                {corr_val} (Spearman)")


    # plot correlation heatmap
    st.subheader('Correlations')
    df_heatmap = df_analysis.copy()
    df_heatmap = df_heatmap[df_heatmap.columns[2:]].corr()
    df_heatmap.columns = [i.capitalize().replace('_', ' ') for i in df_heatmap.columns]
    df_heatmap.index = df_heatmap.columns
    fig_heatmap = get_heatmap(df_heatmap)
    st.plotly_chart(plotly_streamlit_layout(fig_heatmap, height=900, width=900))

elif active_tab == "Covid-19 Time Series Analysis":

    a = st.beta_expander('About (click to expand)')

    a.write("This is an animated plot where each step is a week. If you want to explore the animation using other features \
            wait until the initial animation has ended or skip through to the end before changing features otherwise you may \
            see some odd behaviour in the animation. Plotly can be a little fickle like that.")

    a.write("This is a Plotly chart so you can click on the legend to mask values if needed.")

    # set up some columns for the interactive widgets - use a mid point to create some buffer between widgets
    c1, c2 = st.beta_columns((2, 3))

    # a dictionary of string representation keys and feature values (dependent variables)
    metric_dic = {'Total Deaths': 'total_deaths',
                  'Total Deaths per Million': 'total_deaths_per_million',
                  'Total Cases per Million': 'total_cases_per_million',
                  'People Vaccinated per Hundred': 'people_vaccinated_per_hundred',
                  'Hospital patients per million': 'hosp_patients_per_million',
                  'Population':'population'}



    # user input options for the x axis
    x_metric = c1.selectbox('X axis', ('Total Deaths','Population','Hospital patients per million'),1)

    # user input options for the y axis
    y_metric = c2.selectbox('Y axis', ('People Vaccinated per Hundred','Total Cases per Million',\
                                       'Hospital patients per million'),0)

    # user input options for marker size
    size_by = c1.selectbox('Size markers by', ('Total Deaths', 'Total Deaths per Million', 'Population', \
                                               'People Vaccinated per Hundred'), 1)

    # select which measure of central tendency to show
    average_kind = c2.selectbox('Which central tendency', ('Mean', 'Median'), 0)

    # some custom ranges for different metrics - these are used to make the plot readable
    if x_metric == 'Population':
        range_x = [30000, df_final[metric_dic[x_metric]].max() * 1.4]

    else:
        range_x = [(df_final[metric_dic[x_metric]].min() + 1) * 1.4, df_final[metric_dic[x_metric]].max() * 1.4]

    if y_metric == 'People Vaccinated per Hundred':
        range_y = [-10, 100]
        log_y = False

    else:
        range_y = [(df_final[metric_dic[y_metric]].min() + 1) * 1.4, df_final[metric_dic[y_metric]].max() * 1.4]
        log_y = True

    # main function that generates the animated plot
    def plot_scatter_animate(df, x, y, marker_size, average_kind='Mean'):

        fig = px.scatter(df, x=metric_dic[x_metric], y=metric_dic[y_metric], animation_frame=df.date, \
                         animation_group="location", size=marker_size, hover_name="location", log_y=log_y, log_x=True,
                         color=df.continent, range_x=range_x, range_y=range_y, size_max=50)
        # add vertical and horizontal lines to represent mean or median
        if average_kind == 'Mean':
            fig.add_vline(x=df[x].mean(), line_width=1, line_dash="dash", line_color="grey")
            fig.add_hline(y=df[y].mean(), line_width=1, line_dash="dash", line_color="grey")
        else:
            fig.add_vline(x=df[x].median(), line_width=1, line_dash="dash",
                                  line_color="grey")
            fig.add_hline(y=df[y].median(), line_width=1, line_dash="dash",
                                  line_color="grey")

        # some formatting of plot - axis font size and tick size
        fig.update_layout(yaxis=dict(title=y.capitalize().replace('_', ' '), titlefont_size=15, tickfont_size=10),
                                  xaxis=dict(title=x.capitalize().replace('_', ' '), titlefont_size=15, tickfont_size=10))

        return fig

    fig_animate = plot_scatter_animate(df_final, x=metric_dic[x_metric], y=metric_dic[y_metric], marker_size=\
                                       metric_dic[size_by], average_kind=average_kind)

    fig_animate = plotly_streamlit_layout(fig_animate, height=700, width=1600)

    st.plotly_chart(fig_animate, x_title=metric_dic[x_metric], y_title=metric_dic[y_metric])






