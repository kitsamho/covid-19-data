import numpy as np
import pandas as pd

import plotly.graph_objects as go

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