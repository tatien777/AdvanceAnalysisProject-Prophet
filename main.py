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
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error
import numpy as np


def mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def print_metrics_error(result_train,result_valid):
  err= {}
  err['mae'] = [-1,-1]
  err['mape'] = [-1,-1]
  # print(result_train,result_valid)
  err['mae'][0] = mean_absolute_error(result_train['y'], result_train['yhat'])
  err['mae'][1] = mean_absolute_error(result_valid['y'],result_valid['yhat'])
  st.write("mae without default tune:", err['mae'][0],\
      "mae of manual tune:", err['mae'][1])
#   st.write()
  err['mape'][0] = mean_absolute_percentage_error(result_train['y'], result_train['yhat'])
  err['mape'][1] = mean_absolute_percentage_error(result_valid['y'],result_valid['yhat'])
  st.write("mape without default tune:", err['mape'][0],\
      "mape of manual tune:", err['mape'][1])
  return err

def main():


    st.title('Forecast Demo :chart_with_upwards_trend:')
    data = pd.read_csv('weatherAUS.csv')

    st.subheader('1. Data loading :floppy_disk:')

    st.dataframe(data.head(5))

    adelaide = data.loc[data.Location == 'Adelaide'][['Location','Date','Temp3pm']].reset_index()
    adelaide.drop(['index'],axis=1,inplace=True)
    adelaide['date'] =  pd.to_datetime(adelaide['Date'], format='%d/%m/%Y')
    st.dataframe(adelaide.tail(5))

    df = pd.DataFrame()
    df['ds'] = adelaide.date 
    df['y'] = adelaide['Temp3pm'].values

    st.header("2. Default Parameters 🚫")

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

    st.subheader('2.1 Component ⚙')
    #plot the model forecast chart with component charts in trend and seasonality 
    fig1 = model.plot_components(forecast)
    fig2 = model.plot(forecast)
    st.write(fig1)
    st.write(fig2)

    # Create Training and Test
    train = df.y[:2555]
    test = df.y[2556:] 
    st.subheader('2.2 Prophet result 🔖')


    #define the forecasting model and set the cap
    model = Prophet()
    #define future as the future 365 days' data, and define forecast as predicting the future by the forecasting model
    model.fit(df[:2555])
    future = model.make_future_dataframe(638, freq='D')
    forecastWithoutTune = model.predict(future)
    # Make as pandas series

    fc_series = pd.Series(forecastWithoutTune[2556:]['yhat'], index=test.index)
    lower_series = pd.Series(forecastWithoutTune[2556:]['yhat_lower'], index=test.index)
    upper_series = pd.Series(forecastWithoutTune[2556:]['yhat_upper'], index=test.index)
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
    forecastWithoutTune['y'] = test
    st.subheader('2.2 AIRMA result (2, 1, 1)')
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

    st.header("3. Parameters configuration 🛠️")
    periods_input = 638
    with st.container():
        st.write('In this section you can modify the algorithm settings.')
            
        # with st.expander("Horizon"):
        #     periods_input = st.number_input('Select how many future periods (days) to forecast.',
        #     min_value = 1, max_value = 366,value=90)
        with st.expander("Growth model 📈"):
            st.write('Prophet uses by default a linear growth model.')
            st.markdown("""For more information check the [documentation](https://facebook.github.io/prophet/docs/saturating_forecasts.html#forecasting-growth)""")

            growth = st.radio(label='Growth model',options=['linear',"logistic"]) 

            if growth == 'linear':
                growth_settings= {
                            'cap':1,
                            'floor':0
                        }
                cap=1
                floor=1
                df['cap']=1
                df['floor']=0

            if growth == 'logistic':
                st.info('Configure saturation')

                cap = st.slider('Cap',min_value=0.0,max_value=1.0,step=0.05)
                floor = st.slider('Floor',min_value=0.0,max_value=1.0,step=0.05)
                if floor > cap:
                    st.error('Invalid settings. Cap must be higher then floor.')
                    growth_settings={}

                if floor == cap:
                    st.warning('Cap must be higher than floor')
                else:
                    growth_settings = {
                        'cap':cap,
                        'floor':floor
                        }
                    df['cap']=cap
                    df['floor']=floor
            
            n_changepoints = st.slider('n_changepoints (number of change points)',0,100,25)

        with st.expander("Seasonality 📅"):
            st.markdown("""The default seasonality used is additive, but the best choice depends on the specific case, therefore specific domain knowledge is required. For more informations visit the [documentation](https://facebook.github.io/prophet/docs/multiplicative_seasonality.html)""")
            seasonality = st.radio(label='Seasonality',options=['additive','multiplicative'])

        with st.expander("Seasonality components"):
            st.write("Add or remove components:")
            daily = st.checkbox("Daily")
            weekly= st.checkbox("Weekly")
            monthly = st.checkbox("Monthly")
            yearly = st.checkbox("Yearly")

            
        with st.expander('Holidays 🎎'):
            
            countries = ['Country name','Italy','Spain','United States','France','Germany','Ukraine']
            
            with st.container():
                years=[2021]
                selected_country = st.selectbox(label="Select country",options=countries)

                if selected_country == 'Italy':
                    for date, name in sorted(holidays.IT(years=years).items()):
                        st.write(date,name) 
                            
                if selected_country == 'Spain':
                    
                    for date, name in sorted(holidays.ES(years=years).items()):
                            st.write(date,name)                      

                if selected_country == 'United States':
                    
                    for date, name in sorted(holidays.US(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'France':
                    
                    for date, name in sorted(holidays.FR(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'Germany':
                    
                    for date, name in sorted(holidays.DE(years=years).items()):
                            st.write(date,name)
                            
                if selected_country == 'Ukraine':
                    
                    for date, name in sorted(holidays.UKR(years=years).items()):
                            st.write(date,name)

                else:
                    holidays = False
                            
                holidays = st.checkbox('Add country holidays to the model')

        with st.expander('Scale point 🧰'):
            st.write('In this section it is possible to tune the scaling coefficients.')
            
            seasonality_scale_values= [0.1, 1.0,5.0,50.0]    
            changepoint_scale_values= [0.01, 0.1, 0.5,50]

            st.write("The changepoint prior scale determines the flexibility of the trend, and in particular how much the trend changes at the trend changepoints.")
            changepoint_scale= st.select_slider(label= 'Changepoint prior scale',options=changepoint_scale_values)
            
            st.write("The seasonality change point controls the flexibility of the seasonality.")
            seasonality_scale= st.select_slider(label= 'Seasonality prior scale',options=seasonality_scale_values)    

            st.markdown("""For more information read the [documentation](https://facebook.github.io/prophet/docs/diagnostics.html#parallelizing-cross-validation)""")

    st.subheader("3. Forecast 🪄")
    st.write("Fit the model on the data and generate future prediction.")
    st.write("Load a time series to activate.")

    if input:
        
        if st.checkbox("Initialize model (Fit)",key="fit"):
            if len(growth_settings)==2:
                m = Prophet(seasonality_mode=seasonality,
                            daily_seasonality=daily,
                            weekly_seasonality=weekly,
                            yearly_seasonality=yearly,
                            growth=growth,
                            changepoint_prior_scale=changepoint_scale,
                            seasonality_prior_scale= seasonality_scale,
                            n_changepoints=n_changepoints,
                            )
                if holidays:
                    m.add_country_holidays(country_name=selected_country)
                    
                if monthly:
                    m.add_seasonality(name='monthly', period=30.4375, fourier_order=5)
                    

                with st.spinner('Fitting the model..'):

                    m = m.fit(df[:2555])
                    future = m.make_future_dataframe(periods=periods_input,freq='D')
                    future['cap']=cap
                    future['floor']=floor
                    st.write("The model will produce forecast up to ", future['ds'].max())
                    st.success('Model fitted successfully')

            else:
                st.warning('Invalid configuration')

        if st.checkbox("Generate forecast (Predict)",key="predict"):
            try:
                with st.spinner("Forecasting.."):

                    forecastWithTune = m.predict(future)
                    st.success('Prediction generated successfully')
                    st.dataframe(forecastWithTune)
                    fig1 = m.plot(forecastWithTune)
                    st.write(fig1)
                    output = 1

                    if growth == 'linear':
                        fig2 = m.plot(forecastWithTune)
                        a = add_changepoints_to_plot(fig2.gca(), m, forecastWithTune)
                        st.write(fig2)
                        output = 1
                forecastWithTune['y'] = test
            except:
                st.warning("You need to train the model first.. ")
                    
        
        if st.checkbox('Show components'):
            # try:
            #     with st.spinner("Loading.."):
            #         fig3 = m.plot_components(forecast)
            #         st.write(fig3)
            # except: 
            #     st.warning("Requires forecast generation..") 
            # Make as pandas series

            fc_series = pd.Series(forecastWithTune[2556:]['yhat'], index=test.index)
            lower_series = pd.Series(forecastWithTune[2556:]['yhat_lower'], index=test.index)
            upper_series = pd.Series(forecastWithTune[2556:]['yhat_upper'], index=test.index)
            # Plot
            prophet_plot2 = plt.figure(figsize=(12,5), dpi=100)
            plt.plot(train, label='training')
            plt.plot(test, label='actual')
            plt.plot(fc_series, label='forecast')
            plt.fill_between(lower_series.index, lower_series, upper_series, 
                            color='k', alpha=.15)
            plt.title('Forecast vs Actuals')
            plt.legend(loc='upper left', fontsize=8)
            plt.show()
            st.pyplot(prophet_plot2)



            st.subheader('4. Model validation 🧪')
            st.write("In this section it is possible to do cross-validation of the model.")
            # st.markdown('**Metrics definition**')
            # st.write("Mse: mean absolute error")
            # st.write("Mae: Mean average error")
            # st.write("Mape: Mean average percentage error")
            # st.write("Mse: mean absolute error")
            # st.write("Mdape: Median average percentage error")
            data = {
                'metric' : ['Mean Absolute Error (mae)','Mean Squared Error (mse)','Mean Average Percentage Error (mape)'],
                'Prophet With Default':  [
                mean_absolute_error(test,forecastWithoutTune[2556:]['yhat']),
                mean_squared_error(test,forecastWithoutTune[2556:]['yhat']),
                mean_absolute_percentage_error(test,forecastWithoutTune[2556:]['yhat'])
                ],
                'Prophet With Manual Tune':  [
                mean_absolute_error(test,forecastWithTune[2556:]['yhat']),
                mean_squared_error(test,forecastWithTune[2556:]['yhat']),
                mean_absolute_percentage_error(test,forecastWithTune[2556:]['yhat'])]
                
            }
            metricDF = pd.DataFrame.from_dict(data)
            st.dataframe(metricDF)

    
        
            
main()