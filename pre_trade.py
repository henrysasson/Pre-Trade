import pandas as pd
import numpy as np
import datetime
from datetime import datetime
import time
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='Pre Trade', layout='wide')

st.title('Expected Risk')
st.markdown('##')

def get_data(tickers):
    data = yf.download(tickers, period='5y')['Adj Close']
    return data

def sigma_from_corr_and_std(stdev_list, corrmatrix):
        sigma = np.diag(stdev_list).dot(corrmatrix).dot(np.diag(stdev_list))
        return sigma

# Input para o número de ações no portfólio
number = st.number_input(
    "Insert a number of instruments in the portfolio:", value=1, step=1, min_value=1
)

# Listas para armazenar os tickers, moedas e porcentagens de capital
tickers = []
currency = []
capital_percentages = []

# Criação das colunas dinamicamente
cols = st.columns(number)

# Nomeação das colunas e exibição de um campo de input em cada uma
for i, col in enumerate(cols, 1):
    with col:
        ticker = st.text_input(f"Instrument {i}", placeholder=f"Type the ticker {i}...")
        
        # Verifica se o ticker foi preenchido
        if ticker:
            tickers.append(ticker)  # Adiciona o ticker à lista

            # Tenta obter a moeda do ticker; usa uma alternativa padrão em caso de erro
            try:
                fx = yf.Ticker(ticker).info['currency']
                if fx == 'BRL':
                    currency.append('BRL')
                else:
                    fx = fx + 'BRL=X'
                    currency.append(fx)
            except KeyError:
                st.warning(f"Currency information for ticker {ticker} not available.")
                currency.append('BRL')  # Definindo um valor padrão para evitar erros posteriores
        else:
            currency.append(None)  # Adiciona None para evitar problemas de indexação mais adiante

        # Campo para a porcentagem de capital
        capital_percentage = st.number_input(
            f"% of Capital {i}", format="%.2f"
        )
        capital_percentages.append(capital_percentage)

# Remove entradas vazias
tickers = [ticker for ticker in tickers if ticker]
exposure = [cp for cp in capital_percentages if cp]
currency = [fx for fx in currency if fx]

if tickers and exposure:
    price = get_data(tickers)

    # Verifica se todas as moedas são "BRL" para decidir sobre o ajuste de moeda
    if set(currency) == {"BRL"}:
        # Todas as moedas são BRL, sem ajuste
        adjusted_price = price
    else:
            # Ajuste de moeda necessário
            adjusted_fx = [cur for cur in currency if cur != "BRL"]
            unique_currency = pd.Series(adjusted_fx).unique().tolist()

            fx_price = yf.download(unique_currency, period='5y')['Adj Close']
            fx_price['BRL'] = 1

            # Cria o DataFrame final com as moedas correspondentes para cada ticker
            currency_df = pd.DataFrame(index=fx_price.index, columns=price.columns)
            fx_rate = pd.DataFrame(currency, index=tickers, columns=['FX'])

            for ticker in tickers:
                currency_df[ticker] = fx_price[fx_rate.loc[ticker].values]
                
            # Ajuste de preço com as taxas de câmbio
    
            adjusted_price = currency_df * price

    daily_returns = adjusted_price.pct_change()

    value_of_positions_proportion_capital = pd.DataFrame([exposure], columns=tickers, index = price.index)

    value_of_positions_proportion_capital = value_of_positions_proportion_capital/100

    

    rolling_std = daily_returns.ewm(span=120).std()
    rolling_corr = daily_returns.ewm(span=30).corr()

    expected_risk_df = pd.DataFrame(index = value_of_positions_proportion_capital.index, columns = ['Expected Risk'])

    for index_date in value_of_positions_proportion_capital.index:

        std_dev = rolling_std.loc[index_date].values
        std_dev[np.isnan(std_dev)] = 0.0

        weights = value_of_positions_proportion_capital.loc[index_date].values
        weights[np.isnan(weights)] = 0.0

        cmatrix = rolling_corr.loc[index_date].values
        cmatrix[np.isnan(cmatrix)] = 0.0

        sigma = sigma_from_corr_and_std(std_dev, cmatrix)

        portfolio_variance = weights.dot(sigma).dot(weights.transpose())
        portfolio_std = portfolio_variance ** .5

        annualised_portfolio_std = portfolio_std * 16.0

        expected_risk_df.loc[index_date, 'Expected Risk'] = annualised_portfolio_std



        # Criando o gráfico principal
    fig = go.Figure()

    expected_risk_temp = expected_risk_df.loc[expected_risk_df.index >= price.index]

    expected_risk_temp = expected_risk_df.loc[expected_risk_df['Expected Risk']>0]

    # Adicionando a série de risco padrão
    fig.add_trace(
        go.Scatter(
            x=expected_risk_temp.index,
            y=expected_risk_temp['Expected Risk'].values,
            name='',
            mode='lines'
        )
    )


    # Configurando o layout
    fig.update_layout(
        title_text="Expected Risk",
        xaxis_title="Date",
        yaxis_title="",
        width=1050,  # Largura do gráfico
        height=600,  # Altura do gráfico
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Configurando o rangeslider e o rangeselector no eixo X
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                # dict(count=5, label="5y", step="year", stepmode="backward"), # Descomente se necessário
                dict(step="all")
            ])
        )
    )

    # Configurando o formato dos ticks do eixo Y
    fig.update_yaxes(tickformat=".2%")

    st.markdown('##')

    st.text(f"The current expected risk of the portfolio is: {expected_risk_df.iloc[-1].values[0] * 100:.2f}%")

    # Exibindo o gráfico
    st.plotly_chart(fig, use_container_width=True)

    


else:
    st.text(' ')
# Exibindo a figura
st.plotly_chart(fig)

