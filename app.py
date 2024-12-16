# Import necessary libraries
import holidays
import requests 
import streamlit as st
import pandas as pd

from prophet import Prophet 
from prophet.plot import plot_components_plotly, plot_plotly    #funzionalità nuova per fare i plot dei grafici

# Page configuration
st.set_page_config(page_title="Insect Dashboard", page_icon="🐜")

st.title("Insect Forecasting")
st.info("""
Welcome to the Insect Forecasting Dashboard! 🐜

This application visualizes the forecasting of insect counts using Facebook's **Prophet** library. 
Key features include:
- **Interactive Plotly Graph**: View predictions, confidence intervals, and historical data points.
- **Customizable Forecasting**: Select growth models and configure seasonality to adapt to your dataset.

Prophet helps analyze trends and predict future values with high accuracy. Use this dashboard to explore how environmental factors or historical patterns influence insect population trends.

Enjoy exploring the data! 📊
""")

# Helper function to fetch ticker data with caching
@st.cache_data
def fetch_cicalino_data():
    try:
        cic1_df = pd.read_csv("https://drive.google.com/uc?id=1nrjDiYRusERc0_a86oX_xHMQnMy99zQt")
        cic2_df = pd.read_csv("https://drive.google.com/uc?id=1kAlOrtjwPLUwO7BrQDCcFVc63AwbcM-y")
        cic1_future = pd.read_csv("https://drive.google.com/uc?id=1mN_WgzN1tt4gQIXtTM4tGI7l_hhNESz4")
        cic2_future = pd.read_csv("https://drive.google.com/uc?id=175Yw0UiSkzwaQQoHpIDGQvy3N0hOKi5z")

        if cic1_df.empty or cic2_df.empty:
            st.error(
                f"No data available"
            )
            return None, None, None, None
        else:
            return cic1_df, cic2_df, cic1_future, cic2_future
        
    except Exception as e:
        st.error(f"Error retrieving data for Cicalino")
        return None, None, None, None

@st.cache_data
def fetch_imola_data():
    try:
        imo1_df = pd.read_csv("https://drive.google.com/uc?id=1vQeFWuM2l3SHLohiU_cVfKoBhesa6Z0I")
        imo2_df = pd.read_csv("https://drive.google.com/uc?id=1zDnP1SF2o1iYxL1AzHteG6GAllWSENcN")
        imo1_future = pd.read_csv("https://drive.google.com/uc?id=1H8EzBOOlLELoE5CetGTdg7RtX_OCiOZE")
        imo2_future = pd.read_csv("https://drive.google.com/uc?id=1czeE7a0KvsKpu1W2XrBSm9e81wyHXDfL")
        if imo1_df.empty or imo2_df.empty:
            st.error(
                f"No data available"
            )
            return None, None, None, None
        else:
            return imo1_df, imo2_df, imo1_future, imo2_future
        
    except Exception as e:
        st.error(f"Error retrieving data for Cicalino")
        return None, None, None, None


# Helper function to fit the Prophet model with caching
@st.cache_data 
def fit_prophet_model(
    prophet_df,
    growth,
    seasonality_mode,
    weekly,
    monthly,
    yearly,
    holidays_country,
    daily,
    cap=None,
):
    model = Prophet(
        growth=growth,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        daily_seasonality=daily,
    )
    if holidays_country != "None":
        model.add_country_holidays(country_name=holidays_country)
    if monthly:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    if growth == "logistic" and cap is not None: 
        prophet_df["cap"] = cap
    
    
    model.fit(prophet_df)
    return model

@st.cache_data 
def fit_prophet_model_exogenous(
    prophet_df,
    growth,
    seasonality_mode,
    weekly,
    monthly,
    yearly,
    holidays_country,
    daily,
    cap=None,
):
    model = Prophet(
        growth=growth,
        seasonality_mode=seasonality_mode,
        weekly_seasonality=weekly,
        yearly_seasonality=yearly,
        daily_seasonality=daily,
    )
    if holidays_country != "None":
        model.add_country_holidays(country_name=holidays_country)
    if monthly:
        model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    if growth == "logistic" and cap is not None: 
        prophet_df["cap"] = cap
    
    model.add_regressor("Evento")
    model.add_regressor("Temperatura Minima")
    model.add_regressor("Temperatura Massima")
    model.add_regressor("Media Temperatura")
    model.add_regressor("Media Umidità")

    model.fit(prophet_df)
    return model
    
    

# Sidebar - Ticker selection
st.sidebar.subheader("Parameters")
locations_list = ["Cicalino", "Imola"]
location_selection = st.sidebar.selectbox(
    label="Locations",
    options=sorted(locations_list),
    index=0
)

# Fetch data
if(location_selection == "Cicalino"):
    df_1, df_2, df_1_future, df_2_future = fetch_cicalino_data()
else:
    df_1, df_2, df_1_future, df_2_future = fetch_imola_data()

if df_1 is not None and df_2 is not None:
 
    # Display location data
    if(location_selection == "Cicalino"):
        st.header("Cicalino 1 Data")
        with st.expander("Show Cicalino 1 Data"):
            st.dataframe(df_1[["DateTime", "Numero di insetti", "Evento", "Temperatura Minima", "Temperatura Massima", "Media Temperatura", "Media Umidità"]])
        st.header("Cicalino 2 Data")
        with st.expander("Show Cicalino 2 Data"):
            st.dataframe(df_2[["DateTime", "Numero di insetti", "Evento", "Temperatura Minima", "Temperatura Massima", "Media Temperatura", "Media Umidità"]])
    else:
        st.header("Imola 1 Data")
        with st.expander("Show Imola 1 Data"):
            st.dataframe(df_1[["DateTime", "Numero di insetti", "Evento", "Temperatura Minima", "Temperatura Massima", "Media Temperatura", "Media Umidità"]])
        st.header("Imola 2 Data")
        with st.expander("Show Imola 2 Data"):
            st.dataframe(df_2[["DateTime", "Numero di insetti", "Evento", "Temperatura Minima", "Temperatura Massima", "Media Temperatura", "Media Umidità"]])
    
    # Sidebar - Prophet parameters
    st.sidebar.subheader("Prophet Parameters Configuration")
    horizon_selection = st.sidebar.slider(
        "Forecasting Horizon (days)", min_value=1, max_value=14, value=7
    )
    growth_selection = st.sidebar.radio("Growth", options=["linear", "logistic"])

    # Additional parameters for logistic growth
    cap_close = None
    if growth_selection == "logistic":
        st.sidebar.info(
            "Configure logistic growth saturation as a percentage of latest Close"
        )
        cap = st.sidebar.slider(
            "Carrying Capacity Multiplier", min_value=1.0, max_value=2.0, value=1.2
        )
        cap_close = cap * df_1["Numero di insetti"].iloc[-1]
        df_1["cap"] = cap_close
        cap_close = cap * df_2["Numero di insetti"].iloc[-1]
        df_2["cap"] = cap_close

    seasonality_selection = st.sidebar.radio(
        "Seasonality Mode", options=["additive", "multiplicative"]
    )

    st.sidebar.subheader("Prophet Exogenous Variables")
    exogenous_selection = st.sidebar.checkbox("Exogenous Variables", value=False)

    # Seasonality components
    st.sidebar.subheader("Seasonality Components")
    daily_selection = st.sidebar.checkbox("Daily Seasonability", value=False)
    weekly_selection = st.sidebar.checkbox("Weekly Seasonality", value=False)
    monthly_selection = st.sidebar.checkbox("Monthly Seasonality", value=False)
    yearly_selection = st.sidebar.checkbox("Yearly Seasonality", value=False)

    # Holiday effects
    holiday_country_list = ["None"] + sorted(holidays.list_supported_countries())
    holiday_country_selection = st.sidebar.selectbox(
        "Holiday Country", options=holiday_country_list
    )

    prophet_df_1 = df_1.rename(columns={"DateTime": "ds", "Numero di insetti": "y"})
    if growth_selection == "logistic" and cap_close is not None:
        prophet_df_1["cap"] = cap_close

    prophet_df_2 = df_2.rename(columns={"DateTime": "ds", "Numero di insetti": "y"})
    if growth_selection == "logistic" and cap_close is not None:
        prophet_df_2["cap"] = cap_close

    # Forecasting
    st.header("Forecasting")
    with st.spinner("Fitting the model..."):
        model_1 = fit_prophet_model(
            prophet_df_1,
            growth=growth_selection,
            seasonality_mode=seasonality_selection,
            weekly=weekly_selection,
            monthly=monthly_selection,
            yearly=yearly_selection,
            holidays_country=holiday_country_selection,
            daily=daily_selection,
            cap=cap_close,
        )

    with st.spinner("Fitting the model..."):
        model_2 = fit_prophet_model(
            prophet_df_2,
            growth=growth_selection,
            seasonality_mode=seasonality_selection,
            weekly=weekly_selection,
            monthly=monthly_selection,
            yearly=yearly_selection,
            holidays_country=holiday_country_selection,
            daily=daily_selection,
            cap=cap_close,
        )

    if(exogenous_selection == True):
        with st.spinner("Fitting the model..."):
            model_3 = fit_prophet_model_exogenous(
                prophet_df_1,
                growth=growth_selection,
                seasonality_mode=seasonality_selection,
                weekly=weekly_selection,
                monthly=monthly_selection,
                yearly=yearly_selection,
                holidays_country=holiday_country_selection,
                daily=daily_selection,
                cap=cap_close,
            )

            model_4 = fit_prophet_model_exogenous(
                prophet_df_2,
                growth=growth_selection,
                seasonality_mode=seasonality_selection,
                weekly=weekly_selection,
                monthly=monthly_selection,
                yearly=yearly_selection,
                holidays_country=holiday_country_selection,
                daily=daily_selection,
                cap=cap_close,
            )


    with st.spinner("Generating forecast..."):
        future_1 = model_1.make_future_dataframe(periods=horizon_selection)
        if growth_selection == "logistic" and cap_close is not None:
            future_1["cap"] = cap_close
        forecast_1 = model_1.predict(future_1)

    with st.spinner("Generating forecast..."):
        future_2 = model_2.make_future_dataframe(periods=horizon_selection)
        if growth_selection == "logistic" and cap_close is not None:
            future_2["cap"] = cap_close
        forecast_2 = model_2.predict(future_2) 
                
    if(exogenous_selection == True):
        with st.spinner("Generating forecast..."):
            future_3 = model_3.make_future_dataframe(periods=horizon_selection, freq= 'D', include_history=True)
            df1_copy = prophet_df_1.drop(columns=["y"]).copy()
            df1_copy["ds"] = pd.to_datetime(df1_copy["ds"])
            future_3["ds"] = pd.to_datetime(future_3["ds"])
            df_1_future["ds"] = pd.to_datetime(df_1_future["ds"])
            result_1 = pd.concat([df1_copy, df_1_future], axis=0)

            future_3 = pd.merge(future_3, result_1, on="ds", how="left")  # Unisce le variabili esogene

            #st.dataframe(future_3)
            if growth_selection == "logistic" and cap_close is not None:
                future_3["cap"] = cap_close
            forecast_3 = model_3.predict(future_3)

        with st.spinner("Generating forecast..."):
            future_4 = model_4.make_future_dataframe(periods=horizon_selection, freq= 'D', include_history=True)
            df2_copy = prophet_df_2.drop(columns=["y"]).copy()
            df2_copy["ds"] = pd.to_datetime(df2_copy["ds"])
            future_4["ds"] = pd.to_datetime(future_4["ds"])
            df_2_future["ds"] = pd.to_datetime(df_2_future["ds"])
            result_2 = pd.concat([df2_copy, df_2_future], axis=0)

            future_4 = pd.merge(future_4, result_2, on="ds", how="left")  # Unisce le variabili esogene

            #st.dataframe(future_4)
            if growth_selection == "logistic" and cap_close is not None:
                future_4["cap"] = cap_close
            forecast_4 = model_4.predict(future_4)


    # Interactive Plotly forecast plot
    if(location_selection == "Cicalino"):
        st.subheader("Interactive Forecast Plot Cicalino 1")
    else:
        st.subheader("Interactive Forecast Plot Imola 1")



    fig1 = plot_plotly(model_1, forecast_1)
    for trace in fig1['data']:
        if trace['name'] == 'Actual':
            trace['marker']['color'] = 'red'  # Cambia il colore in rosso
            trace['marker']['size'] = 5 
    st.plotly_chart(fig1)

    if(exogenous_selection == True):
        if(location_selection == "Cicalino"):
            st.subheader("Interactive Forecast Plot Cicalino 1 with Exogenous Variables")
        else:
            st.subheader("Interactive Forecast Plot Imola 1 with Exogenous Variables")


        fig5 = plot_plotly(model_3, forecast_3)
        for trace in fig5['data']:
            if trace['name'] == 'Actual':
                trace['marker']['color'] = 'red'  # Cambia il colore in rosso
                trace['marker']['size'] = 5 
        st.plotly_chart(fig5)


    if(location_selection == "Cicalino"):
        st.subheader("Interactive Forecast Plot Cicalino 2")
    else:
        st.subheader("Interactive Forecast Plot Imola 2")

    fig2 = plot_plotly(model_2, forecast_2)
    for trace in fig2['data']:
        if trace['name'] == 'Actual':
            trace['marker']['color'] = 'red'  # Cambia il colore in rosso
            trace['marker']['size'] = 5 
    st.plotly_chart(fig2)

    if(exogenous_selection == True):
        if(location_selection == "Cicalino"):
            st.subheader("Interactive Forecast Plot Cicalino 2 with Exogenous Variables")
        else:
            st.subheader("Interactive Forecast Plot Imola 2 with Exogenous Variables")
        
        fig6 = plot_plotly(model_4, forecast_4)
        for trace in fig6['data']:
            if trace['name'] == 'Actual':
                trace['marker']['color'] = 'red'  # Cambia il colore in rosso
                trace['marker']['size'] = 5 
        st.plotly_chart(fig6)

    if(location_selection == "Cicalino"):
        with st.expander("Show Forecast Components for Cicalino 1"):
            st.subheader("Forecast Components")
            fig3 = plot_components_plotly(model_1, forecast_1)
            st.plotly_chart(fig3, key = "forecast_components_graph_1")

        with st.expander("Show Forecast Components for Cicalino 2"):
            st.subheader("Forecast Components")
            fig4 = plot_components_plotly(model_2, forecast_2)
            st.plotly_chart(fig4, key = "forecast_components_graph_2")
    else:
        with st.expander("Show Forecast Components for Imola 1"):
            st.subheader("Forecast Components")
            fig3 = plot_components_plotly(model_1, forecast_1)
            st.plotly_chart(fig3, key = "forecast_components_graph_1")

        with st.expander("Show Forecast Components for Imola 2"):
            st.subheader("Forecast Components")
            fig4 = plot_components_plotly(model_2, forecast_2)
            st.plotly_chart(fig4, key = "forecast_components_graph_2")

else:
    st.warning("Please select a different location or time period.")
