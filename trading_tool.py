import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import yfinance as yf
import warnings
import streamlit as st



st.set_page_config(page_title='HS Market Monitor', layout='wide')

st.title('Trading Tool')
    st.markdown('##')


def quantile_of_points_in_data_series (data_series):
    
    data_series = data_series.fillna(0)
    
    numpy_series = np.array(data_series)
    results = []

    for irow in range(len(data_series)):
        current_value = numpy_series[irow]
        count_less_than = (numpy_series < current_value)[:irow].sum()
        results.append(count_less_than / (irow + 1))

    results_series = pd.Series(results, index=data_series.index)
    return results_series


def volatility_regime_multiplier (price):
    
    # Calculate historical Volatility Ratio
    
    daily_returns = price.pct_change()
    
    anual_std = sass.mixed_vol_calc(daily_returns) * 16
     
    ten_year_average = anual_std.rolling(2500, min_periods=10).mean()
    
    normalised_vol = (anual_std / ten_year_average).ewm(span=10).mean()
    
    
    # Define the Volatility Quantile
    if isinstance(price, pd.DataFrame):
        vol_quantile = normalised_vol.apply(quantile_of_points_in_data_series)
    else:
        vol_quantile = quantile_of_points_in_data_series(normalised_vol)
    
    
    # Apply Volatility Regime Multiplier
    
    vol_quantile.fillna(1, inplace = True)
       
    multiplier = 2 - (1.5 * vol_quantile)
    

    return multiplier


def ewmac (price, Lfast, Lslow=None):
    
    # Calculate fast and slow EWMA
    Lslow = 4 * Lfast
    fast_ewma = price.ewm(span=Lfast).mean()
    slow_ewma = price.ewm(span=Lslow).mean()
    raw_ewmac = fast_ewma - slow_ewma
    
    
    # Volatility adjustment
    stdev_returns = sass.mixed_vol_calc(price.diff())
    vol_adj_ewmac = raw_ewmac / stdev_returns
    
    
    # Apply Volatility Regime Multiplier  
    vol_regime_multiplier = volatility_regime_multiplier(price)
    
    raw_forecast_adj = vol_adj_ewmac * vol_regime_multiplier
    
    # Apply Forecast Scalr
    scalar_dict = {64: 1.91, 32: 2.79, 16: 4.1, 8: 5.95, 4: 8.53, 2: 12.1}

    forecast_scalar = scalar_dict[Lfast]
    
    scaled_forecast = raw_forecast_adj * forecast_scalar
    
    capped_forecast = scaled_forecast.clip(-20, 20)
    
    
    return capped_forecast



instrument = st.text_input(label="Choose the instrument (the ticker symbol must be in Yahoo Finance format):", value="")

price = yf.download(instrument, period='25y')['Adj Close']


## EWMAC
ewmac_2_8 = ewmac(price, 2)
ewmac_4_16 = ewmac(price, 4)
ewmac_8_32 = ewmac(price, 8)
ewmac_16_64 = ewmac(price, 16)
ewmac_32_128 = ewmac(price, 32)
ewmac_64_256 = ewmac(price, 64)


combined_forecast = (ewmac_2_8+
ewmac_4_16+
ewmac_8_32+
ewmac_16_64+
ewmac_32_128+
ewmac_64_256) / 6

combined_forecast = combined_forecast * 1.53

combined_forecast = combined_forecast.clip(-20, 20)

combined_forecast = round(combined_forecast, 2)


def forecast_metric(value):
    if 15 < value <= 20:
        reading = 'Very Strong Buy'
    elif 10 < value <= 15:
        reading = 'Strong Buy'
    elif 5 < value <= 10:
        reading = 'Buy'
    elif 0 < value <= 5:
        reading = 'Weak Buy'
    elif -5 < value <= 0:
        reading = 'Neutral'
    elif -10 < value <= -5:
        reading = 'Weak Sell'
    elif -15 < value <= -10:
        reading = 'Sell'
    elif -20 < value <= -15:
        reading = 'Strong Sell'
    elif value <= -20:
        reading = 'Very Strong Sell'
    else:
        reading = 'Invalid Value'

    return reading


# Creating the DataFrame using the specified structure
data_values = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
columns = ["Very strong sell", "Strong Sell", "Sell", "Weak sell", "Neutral", "Weak buy", "Buy", "Strong buy", "Very strong buy"]

# Creating the DataFrame
df = pd.DataFrame([data_values], columns=columns, index=['Values'])

df


# Criando subplots com dois gráficos, um embaixo do outro, e ajustando o espaçamento vertical
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.2)

# Adicionando o traço do Combined Forecast no primeiro gráfico
fig.add_trace(
    go.Scatter(x=combined_forecast.index, y=combined_forecast.values, name="Indicator"),
    row=1, col=1
)

# Adicionando o traço do Price no segundo gráfico
fig.add_trace(
    go.Scatter(x=price.index, y=price.values, name="Price", line=dict(color="green")),
    row=2, col=1
)

# Configurando o layout
fig.update_layout(
    title_text=str(instrument)+" - Indicator and Price",
    xaxis_title="Dates"
)

# Eixo Y do primeiro gráfico (Indicator)
fig.update_yaxes(title_text="Forecast", row=1, col=1)

# Eixo Y do segundo gráfico (Price)
fig.update_yaxes(title_text="Price", row=2, col=1)

# Configurando o layout
fig.update_layout( width=1050,  # Largura do gráfico
                height=900  # Altura do gráfico
            )

# Configurando o rangeslider e o rangeselector no eixo X compartilhado
fig.update_xaxes(
    rangeslider_visible=False,
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(count=3, label="3y", step="year", stepmode="backward"),
            dict(count=5, label="5y", step="year", stepmode="backward"),
            dict(step="all")
        ]))
#     ),
#     tickformat=".2f"
)

fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=0.9
    )
)

# Exibindo a figura
st.plotly_chart(fig)

