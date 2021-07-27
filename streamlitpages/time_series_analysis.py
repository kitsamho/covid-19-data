import streamlit as st
import plotly.express as px
from helperfunctions.helper import *
import os
pd.set_option('display.float_format', lambda x: '%.0f' % x)


def write(df_final):

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
    return



