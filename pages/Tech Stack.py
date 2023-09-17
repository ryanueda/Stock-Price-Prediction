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
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.express as px
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
yf.pdr_override()
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller



## HEADER
st.set_page_config(page_title='Tech Stack', initial_sidebar_state='expanded')
st.markdown('<h1>Tech Stack Stack</h1>', unsafe_allow_html=True)
st.subheader("Tech Stocks Listed Under SGX")
st.header('')


## SIDEBAR
analytics = st.sidebar.radio('Analytics', options=['Run Sequence (Close)', 'Daily Returns (AVG)', 'Correlation Heatmap', 'Investment Risk Analysis','Moving Averages'])



## ! PROCESSING
@st.cache_data
def processing(tech_list, company_names):

    end = datetime.now()
    start = datetime(end.year - 10, end.month, end.day)

    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)
        

    company_list = [AAPL, GOOG, MSFT, AMZN]

    for company, com_name in zip(company_list, company_names):
        company["company_name"] = com_name
        
    df = pd.concat(company_list, axis=0)

    ## ? DAILY RETURNS
    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()

    ## ? CORRELATION
    closing_df = pdr.get_data_yahoo(tech_list, start=start, end=end)['Adj Close']
    tech_rets = closing_df.pct_change()
    closing_price_corr = closing_df.corr()
    stock_return_corr = tech_rets.corr()

    ## ? INVESTMENT RISK
    rets = tech_rets.dropna()
    area = np.pi * 20

    return company_list, closing_df, tech_rets, closing_price_corr, stock_return_corr, rets, area

# The tech stocks we'll use for this analysis
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
company_names = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]
company_list, closing_df, tech_rets, closing_price_corr, stock_return_corr, rets, area = processing(tech_list, company_names)


## ! RUN SEQUENCE PLOT
if analytics == 'Run Sequence (Close)':
    st.markdown('<h4>Run Sequence Plot (Close)</h4>', unsafe_allow_html=True)
    ## closing price trend lines
    col1, col2 = st.columns(2)
    for idx, company in enumerate(company_list):
        if idx%2 != 0:
            with col1:
                fig = px.line(company['Close'], title=company_names[idx])
                fig.update_xaxes(title_text='Date')
                fig.update_yaxes(title_text='Closing Price')   
                st.plotly_chart(fig, use_container_width=True, sharing='streamlit')
        else:
            with col2:
                fig = px.line(company['Close'], title=company_names[idx])
                fig.update_xaxes(title_text='Date')
                fig.update_yaxes(title_text='Closing Price')   
                st.plotly_chart(fig, use_container_width=True, sharing='streamlit')


## ! DAILY RETURNS (AVG)
if analytics == 'Daily Returns (AVG)':
    st.markdown('<h4>Daily Returns (AVG)</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for idx, company in enumerate(company_list):
        if idx%2 != 0:
            with col1:
                fig = px.line(company, x=company.index, y='Daily Return', title=company_names[idx])
                fig.update_traces(mode='lines+markers', line=dict(dash='dash'), marker=dict(symbol='circle', size=8))
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            with col2:
                fig = px.line(company, x=company.index, y='Daily Return', title=company_names[idx])
                fig.update_traces(mode='lines+markers', line=dict(dash='dash'), marker=dict(symbol='circle', size=8))
                fig.update_layout(showlegend=True)
                st.plotly_chart(fig, use_container_width=True)

    ## TODO: DISTRIBUTION PLOT
    st.markdown('<h4>Distribution Plot</h4>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    for idx, company in enumerate(company_list):
        if idx%2 != 0:
            with col1:
                fig2 = px.histogram(company['Daily Return'], title=company_names[idx])
                st.plotly_chart(fig2, use_container_width=True)
        else:
            with col2:
                fig2 = px.histogram(company['Daily Return'], title=company_names[idx])
                st.plotly_chart(fig2, use_container_width=True)


## ! CORRELATION HEATMAP
if analytics == 'Correlation Heatmap':
    fig = px.imshow(stock_return_corr,
                    color_continuous_scale='RdBu',
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    title="Correlation Heatmap")

    # Create annotations for correlation values
    annotations = []
    for i in range(len(stock_return_corr)):
        for j in range(len(stock_return_corr)):
            annotations.append(
                dict(
                    x=stock_return_corr.columns[i],
                    y=stock_return_corr.columns[j],
                    text=str(round(stock_return_corr.iloc[i, j], 2)),
                    showarrow=False,
                    font=dict(color='white')
                )
            )

    # Add annotations to the heatmap
    fig.update_layout(annotations=annotations)
    fig.update_layout(
        width=1000,
        height=800,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
    )

    st.plotly_chart(fig, use_container_width=True)


## ! INVESTMENT RISK
if analytics == 'Investment Risk Analysis':
    import plotly.graph_objects as go
    # Create a scatter plot
    scatter_trace = go.Scatter(
        x=rets.mean(),
        y=rets.std(),
        mode='markers',
        marker=dict(
            size=30,
            color='purple',  # You can specify the color here
        ),
    )

    annotations = [
        dict(
            x=x,
            y=y,
            xref='x',
            yref='y',
            text=label,
            showarrow=True,
            arrowhead=7,
            ax=50,
            ay=50,
        )
        for label, x, y in zip(rets.columns, rets.mean(), rets.std())
    ]

    # Create the layout
    layout = go.Layout(
        title="Scatter Plot with Annotations",
        xaxis=dict(title="Expected return"),
        yaxis=dict(title="Risk"),
    )

    # Create the figure
    fig = go.Figure(data=[scatter_trace], layout=layout)
    fig.update_layout(annotations=annotations)

    st.plotly_chart(fig, use_container_width=True)

    recent = pd.to_datetime('2021-01-01')
    risk = pd.DataFrame(columns=['Min', 'Max', 'Mean', 'Variance', 'Standard Deviation', 'Ad-Fuller Value', 'Stationarity', 'Growth Trends', 'Loss Trends', 'Longest Growth', 'Longest Loss'])

    for idx, company in enumerate(company_list):

        recent_data = company[company.index >= recent]

        var = recent_data['Close'].var()
        std = recent_data['Close'].std()
        mean = recent_data['Close'].mean()
        min = recent_data['Close'].min()
        max = recent_data['Close'].max()
        adf = round(adfuller(recent_data['Close'])[1], 2)
        if adf > 0.05:
            stationary = 'False'
        else:
            stationary = 'True'

        
        close = company['Close'].tolist()
        up_trends = 0
        down_trends = 0
        current_trend = None

        # Initialize variables to track the longest up and down trends
        longest_up_trend = 0
        longest_down_trend = 0

        # Initialize variables to count the current up and down trends
        current_up_trend = 0
        current_down_trend = 0

        for i in range(1, len(close)):
            if close[i] > close[i - 1]:  # If the current value is greater than the previous value
                current_up_trend += 1
                current_down_trend = 0  # Reset the down trend counter
                if current_trend != "up":  # If the current trend is not already an up trend
                    up_trends += 1
                    current_trend = "up"
            elif close[i] < close[i - 1]:  # If the current value is less than the previous value
                current_down_trend += 1
                current_up_trend = 0  # Reset the up trend counter
                if current_trend != "down":  # If the current trend is not already a down trend
                    down_trends += 1
                    current_trend = "down"
            else:  # If the current value is equal to the previous value
                current_up_trend = 0
                current_down_trend = 0

            # Update the longest trends
            if current_up_trend > longest_up_trend:
                longest_up_trend = current_up_trend
            if current_down_trend > longest_down_trend:
                longest_down_trend = current_down_trend


        risk.loc[company_names[idx]] = [min, max, mean, var, std, adf, stationary, up_trends, down_trends, longest_up_trend, longest_down_trend]
    st.dataframe(risk, use_container_width=False)


## ! MOVING AVERAGES
if analytics == 'Moving Averages':

    for idx, company in enumerate(company_list):
        windows = [5, 10, 15, 20, 30, 60]
        for window in windows:
            company[f'MA_{window}'] = company['Close'].rolling(window=window).mean()

        # Create a line chart with Plotly Express
        fig = px.line(company, x=company.index, y=company.columns[-len(windows):], labels={'variable': 'Moving Average'})
        fig.update_layout(title=company_names[idx], xaxis_title='Date', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)