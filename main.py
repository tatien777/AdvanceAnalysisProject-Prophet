import streamlit as st 
import pandas as pd 
import matplotlib.pyplot as plt
import datetime as datetime
from prophet import Prophet
import plotly.graph_objects as go
import plotly.figure_factory as ff 
# from fbprophet.plot import plot_plotly, plot_components_plotly
import warnings
import plotly.express as px

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA


data = pd.read_csv('weatherAUS.csv')


st.dataframe(data.head(5))

adelaide = data.loc[data.Location == 'Adelaide'][['Location','Date','Temp3pm']].reset_index()
adelaide.drop(['index'],axis=1,inplace=True)
adelaide['date'] =  pd.to_datetime(adelaide['Date'], format='%d/%m/%Y')
st.dataframe(adelaide.tail(5))

df = pd.DataFrame()
df['ds'] = adelaide.date 
df['y'] = adelaide['Temp3pm'].values



fig = px.line(adelaide, x='date', y="Temp3pm",height=500,width=1200)
fig.update_xaxes(dtick="M2")
fig.update_layout(xaxis=dict(tickformat="%b-%y"))
st.plotly_chart(fig,use_container_width=True)


#define the forecasting model and set the cap
model = Prophet()
#define future as the future 365 days' data, and define forecast as predicting the future by the forecasting model
model.fit(df)
future = model.make_future_dataframe(365, freq='D')
forecast = model.predict(future)

#plot the model forecast chart with component charts in trend and seasonality 
fig1 = model.plot_components(forecast)
fig2 = model.plot(forecast)
st.write(fig1)
st.write(fig2)

# Create Training and Test
train = df.y[:2555]
test = df.y[2556:] 

# Build Model
# model = ARIMA(train, order=(3,2,1))  
# model = sm.tsa.arima.ARIMA(train.dropna(), order=(2, 1, 1))  
model = ARIMA(train.dropna(), order=(2, 1, 1))  
fitted = model.fit()  

# Forecast
fc = fitted.forecast(637, alpha=0.05)  # 95% conf
conf_ins = fitted.get_forecast(637).summary_frame()

# st.write(result)
# fc, se, conf = fitted.forecast(637, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf_ins['mean_ci_lower'], index=test.index)
upper_series = pd.Series(conf_ins['mean_ci_upper'], index=test.index)

# Plot
airma_plot = plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
# plt.show()
st.pyplot(airma_plot)

#define the forecasting model and set the cap
model = Prophet()
#define future as the future 365 days' data, and define forecast as predicting the future by the forecasting model
model.fit(df[:2555])
future = model.make_future_dataframe(640, freq='D')
forecast = model.predict(future)
# Make as pandas series

fc_series = pd.Series(forecast[2556:]['yhat'], index=test.index)
lower_series = pd.Series(forecast[2556:]['yhat_lower'], index=test.index)
upper_series = pd.Series(forecast[2556:]['yhat_upper'], index=test.index)
# Plot
prophet_plot = plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

st.pyplot(prophet_plot)