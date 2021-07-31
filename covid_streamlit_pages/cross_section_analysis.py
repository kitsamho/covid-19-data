import streamlit as st
import plotly.express as px
from sklearn.preprocessing import normalize
from helperfunctions.helper import *
pd.set_option('display.float_format', lambda x: '%.0f' % x)

def write(df):

    a = st.beta_expander('About (click to expand)')

    a.write("A crossplot allows users to plot two features against one another with markers, marker colour and marker sizes \
    representing third and fourth features.")
    a.write("DataFrame masking allows users to explore the data by continent or include all countries.")
    a.write("Each feature's central tendency is represented by a dashed line on each axis so users can see where \
        countries are positioned in terms of the distribution for any given feature.")

    a.write("This is a Plotly chart so users can click on the legend to mask values if needed.")
    a.write('The heat map is a summary of correlations between features.')

    # DataFrame set up
    df_country_unique = df[df.columns[1:3]].drop_duplicates(subset=['location'], keep='first').set_index(
        'location')
    df_analysis = pd.concat([df_country_unique, df.groupby('location')[df.columns[3:]].max()], axis=1)
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
                    'Diabetes Prevalance' :'diabetes_prevalence',
                    'Female Smokers' :'female_smokers',
                    'Male Smokers': 'male_smokers',
                    'Handwashing Facilities': 'handwashing_facilities',
                    'Hospital Beds per Thousand' :'hospital_beds_per_thousand',
                    'Hospital patients per million' :'hosp_patients_per_million',
                    'Life Expectancy': 'life_expectancy'}

    # a dictionary of string representation keys and feature values (dependent variables)
    size_dic = {'Total Deaths': 'total_deaths',
                'Total Deaths per Million': 'total_deaths_per_million',
                'Total Cases per Million': 'total_cases_per_million',
                'People Vaccinated per Hundred': 'people_vaccinated_per_hundred',
                'No sizing': 'no_sizing'}

    # set up some columns for the interactive widgets - use a mid point to create some buffer between widgets
    c1, mid, c2, = st.beta_columns((3 ,0.5 ,3))

    # set up the widget options for continents
    continents = list(df_analysis.continent.unique())
    continents.sort()
    continents.insert(0 ,'Show All Countries')

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
        marker_size = st.slider('Use this to adjust relative marker size', 1, 1000, step=1 ,value=150)
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
        fig_scatter.update_layout(
                                  width=1000,
                                  height=1000,
                                  showlegend=True)

        # some formatting of plot - axis font size and tick size
        fig_scatter.update_layout(yaxis=dict(title=y.capitalize().replace('_', ' '), titlefont_size=15, tickfont_size=10),
                                  xaxis=dict(title=x.capitalize().replace('_' ,' '), titlefont_size=15, tickfont_size=10))

        return fig_scatter


    # get plot
    fig_cross_plot = plot_scatter(df_plot, x=x, y=y, size=size_by, rel_sizing=rel_sizing ,marker_size=marker_size)

    # add plot to streamlit
    st.plotly_chart(plotly_streamlit_layout(fig_cross_plot, height=800, width=1200), use_container_width=True)

    # Show Spearman's correlation co-efficient of variables chosen
    df_corr = df_plot[[x ,y]].corr()
    corr_val = str("{:.2f}".format(df_corr[df_corr.columns[1]].iloc[0]))

    st.markdown(f"Correlation of **{x.capitalize().replace('_' ,' ')}** and **{y.capitalize().replace('_' ,' ')}** = \
                    {corr_val} (Spearman)")


    # plot correlation heatmap
    st.subheader('Correlations')
    df_heatmap = df_analysis.copy()
    df_heatmap = df_heatmap[df_heatmap.columns[2:]].corr()
    df_heatmap.columns = [i.capitalize().replace('_', ' ') for i in df_heatmap.columns]
    df_heatmap.index = df_heatmap.columns
    fig_heatmap = get_heatmap(df_heatmap)
    st.plotly_chart(plotly_streamlit_layout(fig_heatmap, height=900, width=900), use_container_width=True)
    return