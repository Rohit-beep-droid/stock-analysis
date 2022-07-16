"""
This is my first ever stock related application
Streamlit, Prophet (formetly Facebook Prophet), and
Plotly are fairly new to me. But I wanted to get familiar with
them as a tool to learn some new things and build a strong
Python background.

"""

# necessary imports to predict, graph, and analyze market data
import yfinance as y_finance
import datetime as date
import json
import streamlit as streamlit
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as graph

with open('tickers.json', 'r') as tickers:
    data = json.load(tickers)

symbols = data['tickers']

# timeline setup
PAST = "2016-01-01"
CURRENT = date.datetime.now().strftime("%Y-%m-%d")

streamlit.title("Stock Analysis Application")
stock = streamlit.selectbox("Choose a stock:", symbols)
time_period = streamlit.slider("Prediction Timeline:", 1, 10)
time = time_period * 365

# function to find the current stock price
@streamlit.cache
def stock_price(symbol):
    price = y_finance.Ticker(symbol)
    price = price.info['regularMarketPrice']
    return price

# function to download data from yahoo finance
@streamlit.cache
def trend(symbol):
    period = y_finance.download(symbol, PAST, CURRENT)
    period.reset_index(inplace=True)
    return period

status = streamlit.text("Fetching Data...")
current_price = stock_price(stock)
data_timeline = trend(stock)
status.text("Data Retrieved!")
streamlit.subheader("Stock Price:")

# font customization (personal preferences)
streamlit.markdown("""
<style>
.big-font {
    font-size:26px !important;
    font-family: system-ui !important;
    color: rgb(26, 148, 49) !important;
}
</style>
""", unsafe_allow_html=True)
streamlit.markdown(f'<p class="big-font">${current_price}</p>', unsafe_allow_html=True)

def plot():
    # opening price graph
    figure1 = graph.Figure()
    figure1.add_trace(graph.Scatter(x=data_timeline['Date'], y=data_timeline['Open'],
    fill='tonexty', fillcolor='rgba(26, 148, 49, 0.1)', line_color='rgb(26, 148, 49)'))
    figure1.layout.update(title_text="Opening Stock Price Overtime:", xaxis_title='TIME', yaxis_title='PRICE', xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(figure1)

    # closing price graph
    figure2 = graph.Figure()
    figure2.add_trace(graph.Scatter(x=data_timeline['Date'], y=data_timeline['Close'],
    fill='tonexty', fillcolor='rgba(254, 77, 77, 0.1)', line_color='rgb(254, 77, 77, 0.1)'))
    figure2.layout.update(title_text="Closing Stock Price Overtime:", xaxis_title='TIME', yaxis_title='PRICE', xaxis_rangeslider_visible=True)
    streamlit.plotly_chart(figure2)

# plotting the data
plot()

# prediction model using Prophet (formerly FaceBook prophet)
catagories = data_timeline[['Date', 'Close']]
catagories = catagories.rename(columns={"Date": "ds", "Close": "y"})
prophet = Prophet()
prophet.fit(catagories)
prediction = prophet.make_future_dataframe(periods=time)
forecast = prophet.predict(prediction)
streamlit.subheader('Prediction Data:')

# plot graphs generated via plotly
figure1 = plot_plotly(prophet, forecast, trend=True)
figure2 = plot_components_plotly(prophet, forecast)
streamlit.plotly_chart(figure1)
streamlit.subheader("Prediction Data of Components:")
streamlit.plotly_chart(figure2)