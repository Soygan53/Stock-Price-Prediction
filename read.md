---
jupyter:
  colab:
    collapsed_sections:
    - hVqEphVxfYII
    - gFWTxsWvuiiF
    - lpVQ5b0C0QL9
    - \_Pm1Ukgb5sE8
    - 3RjZ8LX86cFM
    - iaipWLBF6jV2
    - DiKiAkRjkD6y
    - UQiBaVKw2q-3
    - UC4qT4y3rbqo
    - rCwhX7ocrfqv
  gpuClass: standard
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

::: {.cell .markdown id="SMrTWgRwQaS6"}
\#\#Summary

One of the most common investment strategies used nowadays appears to be
stock market trades. In this work, price prediction and trading on stock
market transactions are done using systems that use an algorithm that
has been reinforcement learning trained.

These stock market transactions show a time series issue that was
created utilizing the day\'s end closing data. The closing data will be
the main focus of this research, which compares the outcomes of
prediction models on time series and examines some outcomes through
purchasing and selling. Social and economic aspects that could affect
these price models will be left out of the analysis process.

The second section will involve the creation of an agent and the
measurement of success in accordance with various scenarios through the
execution of buy, sell, and hold transactions on the time series. Moving
averages and other indicators will be used to provide the required buy
and sell signals for trading. In the section that follows, titled
\"Conclusion,\" the outcomes of these processes will be discussed.
:::

::: {.cell .markdown id="RGcm9goxVwNP"}
\#\#Motivation Creating the best forecasting model for stock market
transactions is very impossible. This is because the system\'s variables
have an impact on trade values when using blunt data. The data we have,
however, can be used to create models that will allow us to get good
results in this system.

The requirement to create a beautiful algorithm using a limited dataset
is the aspect of this project that most interested me. Years of battling
the causality of my stock market purchasing and selling decisions
prompted me to finally ask, \"I wonder what would happen if an
analytical approach were used to these transactions?\" I began looking
for the response to the query.

I had the possibility to experiment different dynamics and reinforcement
learning while working on this project. He provided me the opportunity;

I want to thank my instructors Tuna Akar, Cem Yiman, and Alperen Sayer
as well as the Coşkunöz Education Foundation and Bursa Eskişehir Bilecik
Development Agency.
:::

::: {.cell .markdown id="Xmd3btV3W0C4"}
\#\#Literature Review

In Diler\'s (2003) study, an artificial neural network model was
employed to try and anticipate the direction of the BIST 100 index
returns one day in advance. In the study, it had a 60.81% accuracy rate
in predicting the price direction.

In their study from 2004, Tektaş and Karataş used artificial neural
networks to estimate the stock prices of companies engaged in the food
and cement industries. The correlation coefficient served as the
measurement of prediction accuracy in their studies. The performance of
artificial neural networks was shown to be superior than the regression
approach at the conclusion of the investigation.

In their study, Karaatl et al. (2005) attempted to forecast the BIST
index\'s closing price. Regression and artificial neural networks were
used to estimate data, and it was shown that artificial neural networks
produced superior outcomes to regression. Interest rates, the price of
gold, the rate of inflation, the index of industrial production, the
interest rate on savings deposits, and the dollar rate were used as
inputs, and monthly forecasts were made. As a performance indicator,
RMSE (Root Mean Squared Error) was utilized.

In their study, Yıldız et al. (2008). employed artificial neural
networks to forecast the direction of the BIST 100 index the following
day. Inputs included variables like the highest and lowest price,
closing price, and TL-Dollar rate. The index direction a day later is
the output variable. 100 observations were utilized to compare the
performance of the model, whereas 1805 observations were used to train
the neural network. In their studies, they had a prediction accuracy of
54.37%.

In their study, Öz et al. (2011) forecasted the returns of the stocks
included in the BIST 30 index in their analysis. With the aid of
discriminant analysis, stock returns were approximated using financial
ratios from one and two years prior. The analysis\'s findings showed
that the model employing the ratios from two years ago had a greater
success rate.

In the study written up by Kara et al. (2011), they used support vector
machines and artificial neural networks to estimate the BIST100 index.
There are ten technical indicators used. The scale for the data set is
\[-1.0, 1.0\]. They found that support vector machines could predict
accurately 71.52% of the time, compared to artificial neural networks\'
average prediction accuracy of 75.74%.
:::

::: {.cell .markdown id="hVqEphVxfYII"}
\#Dataset & Preparation
:::

::: {.cell .code execution_count="4" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="3bC1IoOnfgyx" outputId="6904f086-8567-4ecd-bb23-994be042c775"}
``` {.python}
#!pip install yfinance --upgrade --no-cache-dir
#!pip install pmdarima
#!pip3 install statsmodels

import pandas as pd
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime
import pkg_resources
import types
import time
import numpy as np
import matplotlib.pyplot as plt
from pmdarima.arima import ADFTest,auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

yf.pdr_override()


Akfen = pdr.get_data_yahoo("AKFGY.IS", start='2018-12-01', end='2022-12-13')
Sekerbank = pdr.get_data_yahoo("SKBNK.IS", start='2018-12-01', end='2022-12-13')
Eregli = pdr.get_data_yahoo("EREGL.IS", start='2018-12-01', end='2022-12-13')
Sise = pdr.get_data_yahoo("SISE.IS", start='2018-12-01', end='2022-12-13')
KozaAltin = pdr.get_data_yahoo("KOZAA.IS", start='2018-12-01', end='2022-12-13')
Vestel = pdr.get_data_yahoo("VESTL.IS", start='2018-12-01', end='2022-12-13')
```

::: {.output .stream .stdout}
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
:::
:::

::: {.cell .code execution_count="5" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="5L0C2q7-f87a" outputId="b27a1b94-9200-4037-f279-506acc48d9ee"}
``` {.python}
Akfen_close=Akfen["Close"]
Sekerbank_close=Sekerbank["Close"]
Eregli_close=Eregli["Close"]
Sise_close=Sise["Close"]
KozaAltin_close=KozaAltin["Close"]
Vestel_close=Vestel["Close"]

Akfen_close = Akfen_close.reset_index(name='Akfen')
Sekerbank_close = Sekerbank_close.reset_index(name='Sekerbank')
Eregli_close = Eregli_close.reset_index(name='Eregli')
Sise_close = Sise_close.reset_index(name='Sise')
KozaAltin_close = KozaAltin_close.reset_index(name='KozaAltin')
Vestel_close = Vestel_close.reset_index(name='Vestel')

Hisseler=pd.concat([Akfen_close["Akfen"], Sekerbank_close["Sekerbank"],Eregli_close["Eregli"],
           Sise_close["Sise"],KozaAltin_close["KozaAltin"],Vestel_close["Vestel"]], axis=1)
Hisseler.info()
```

::: {.output .stream .stdout}
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1021 entries, 0 to 1020
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Akfen      1021 non-null   float64
     1   Sekerbank  1021 non-null   float64
     2   Eregli     1021 non-null   float64
     3   Sise       1021 non-null   float64
     4   KozaAltin  1021 non-null   float64
     5   Vestel     1021 non-null   float64
    dtypes: float64(6)
    memory usage: 48.0 KB
:::
:::

::: {.cell .markdown id="gFWTxsWvuiiF"}
\#Exploratory Data Analysis of the Preprocessed Dataset
:::

::: {.cell .code execution_count="6" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="2H_S-bQYujuQ" outputId="a2866224-e63a-431e-c347-34265d5211d4"}
``` {.python}
Son_1_Hafta=Hisseler.iloc[len(Hisseler)-7:len(Hisseler)]
Son_1_Yil=Hisseler.iloc[len(Hisseler)-365:len(Hisseler)]
Degisim_1_Hafta=Son_1_Hafta.pct_change()
Degisim_1_Yil=Son_1_Yil.pct_change()
Degisim_ortalaması_1Hafta=Degisim_1_Hafta.mean(axis=0)
Degisim_ortalaması_1Yil=Degisim_1_Yil.mean(axis=0)
print(Degisim_ortalaması_1Hafta*100)
print(Degisim_ortalaması_1Yil*100)
```

::: {.output .stream .stdout}
    Akfen       -0.084442
    Sekerbank    0.968864
    Eregli       0.036343
    Sise         0.581965
    KozaAltin    2.184173
    Vestel       0.287379
    dtype: float64
    Akfen        0.378143
    Sekerbank    0.327499
    Eregli       0.265593
    Sise         0.486839
    KozaAltin    0.428251
    Vestel       0.248880
    dtype: float64
:::
:::

::: {.cell .code execution_count="7" colab="{\"height\":2339,\"base_uri\":\"https://localhost:8080/\"}" id="hykU1ltWus6I" outputId="d9cad95d-16cb-4294-eb3c-29a1cd7dc413"}
``` {.python}
fig, Koza_fig = plt.subplots(figsize=(8, 6))
plt.ylabel('Koza Altin Lot Değeri(TL)')
plt.xlabel('Gün')
plt.title("Koza Altin 3 Yıllık Grafiği")
fig, Akfen_fig = plt.subplots(figsize=(8, 6))
plt.ylabel('Akfen Lot Değeri(TL)')
plt.xlabel('Gün')
plt.title("Akfen 3 Yıllık Grafiği")
fig, Sekerbank_fig = plt.subplots(figsize=(8, 6))
plt.ylabel('Sekerbank Lot Değeri(TL)')
plt.xlabel('Gün')
plt.title("Sekerbank 3 Yıllık Grafiği")
fig, Eregli_fig = plt.subplots(figsize=(8, 6))
plt.ylabel('Eregli Lot Değeri(TL)')
plt.xlabel('Gün')
plt.title("Eregli 3 Yıllık Grafiği")
fig, Sise_fig = plt.subplots(figsize=(8, 6))
plt.ylabel('Sise Lot Değeri(TL)')
plt.xlabel('Gün')
plt.title("Sise 3 Yıllık Grafiği")
fig, Vestel_fig = plt.subplots(figsize=(8, 6))
plt.ylabel('Vestel Lot Değeri(TL)')
plt.xlabel('Gün')
plt.title("Vestel 3 Yıllık Grafiği")
Koza_fig.plot(KozaAltin["Close"],color="blue")
Akfen_fig.plot(Akfen["Close"],color="blue")
Sekerbank_fig.plot(Sekerbank["Close"],color="blue")
Eregli_fig.plot(Eregli["Close"],color="blue")
Sise_fig.plot(Sise["Close"],color="blue")
Vestel_fig.plot(Vestel["Close"],color="blue")
plt.show()
```

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/f2362b18743722a2f97c2a392ac67bd015cca655.png)
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/c2a94a30e88bb680b9e0076bb98980e3cf6416a6.png)
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/454724fad6872444f5265371a362ddc5af0742d2.png)
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/8a2de921cd7162dfac9d549c76753893971a73d9.png)
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/e5d68c90fcda361b1c5808c79fc632a469379289.png)
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/460d6f7c5d682926164f5e89c5937da7ffbd698f.png)
:::
:::

::: {.cell .code execution_count="8" colab="{\"height\":295,\"base_uri\":\"https://localhost:8080/\"}" id="T9Z-VhmIyk8M" outputId="54618956-ef21-49ce-a5c2-f586cba2dc54"}
``` {.python}
y_pos =["Akfen","Sekerbank","Eregli","Sise","KozaAltin","Vestel"]
# Create bars
plt.title("1 Haftalık ve 1 Yıllık Değişimler")
plt.ylabel('Değişim oranları(%)')
plt.xlabel('Hisse İsimleri')
plt.bar(y_pos,Degisim_ortalaması_1Hafta*100,label="1Hafta",alpha=0.5)
plt.bar(y_pos,Degisim_ortalaması_1Yil*100,label="1Yıl",alpha=0.7)
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/545e2666995dcf1bfbdf600368209d5acc785922.png)
:::
:::

::: {.cell .markdown id="lpVQ5b0C0QL9"}
\#Model Development
:::

::: {.cell .markdown id="_Pm1Ukgb5sE8"}
\#\#Using ARIMA to Predict on Stock Market
:::

::: {.cell .code execution_count="9" id="j_7V97071-en"}
``` {.python}
Hisse=KozaAltin["Close"]
df1 = pd.DataFrame(data=Hisse.index, columns=['Date'])
df2 = pd.DataFrame(data=Hisse.values, columns=['Value'])
df = pd.merge(df1, df2, left_index=True, right_index=True)
train = Hisse[Hisse.index < pd.to_datetime("2021-11-01", format='%Y-%m-%d')]
df1_train = pd.DataFrame(data=train.index, columns=['Date'])
df2_train = pd.DataFrame(data=train.values, columns=['Value'])
df_train = pd.merge(df1_train, df2_train, left_index=True, right_index=True)
test = Hisse[Hisse.index > pd.to_datetime("2021-11-01", format='%Y-%m-%d')]
df1_test = pd.DataFrame(data=test.index, columns=['Date'])
df2_test = pd.DataFrame(data=test.values, columns=['Value'])
df_test = pd.merge(df1_test, df2_test, left_index=True, right_index=True)
```
:::

::: {.cell .code execution_count="10" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="fYK3ECOY57U4" outputId="bd959adb-e0f7-4a05-d237-3bfb8b3a3401"}
``` {.python}
arima_model = auto_arima(df_train["Value"],start_p = 2 ,d =2,start_q=2,max_p=5,max_q=5,n_fits=100)
arima_model
```

::: {.output .execute_result execution_count="10"}
    ARIMA(order=(5, 2, 0), scoring_args={}, suppress_warnings=True,
          with_intercept=False)
:::
:::

::: {.cell .code execution_count="11" colab="{\"height\":322,\"base_uri\":\"https://localhost:8080/\"}" id="v8PBsBxU6C4O" outputId="b3b53988-40da-43f1-d61d-be05b5450601"}
``` {.python}
y = df_train['Value']
ARMAmodel = SARIMAX(y,order = (5, 2, 0), scoring_args={}, suppress_warnings=True,
      with_intercept=False)
ARMAmodel = ARMAmodel.fit()
y_pred = ARMAmodel.get_forecast(len(test.index))
y_pred_df = y_pred.conf_int(alpha = 0.05) 
y_pred_df["Predictions"] = ARMAmodel.predict(start = y_pred_df.index[0], end = y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df["Predictions"]
plt.plot(y_pred_out, color='green', label = 'Predictions')
plt.plot(train, color = "black",label="Train data")
plt.plot(test, color = "red", label="Test data")
plt.ylabel('Koza Altin Price')
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.title("Train/Test split for Koza Altin")
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/32aa70d2ab71899a2ad223526a576b397e8d3809.png)
:::
:::

::: {.cell .code execution_count="12" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="46h1c7xN6KEC" outputId="d4b00406-e6b1-47d1-e289-a752490c07c2"}
``` {.python}
A=y_pred_out.to_frame()
B = A.reset_index(drop=True)
df_new=pd.concat([df_test, B], axis=1)
r2_score(df_new["Value"],df_new["Predictions"])
```

::: {.output .execute_result execution_count="12"}
    -3.604755202101951
:::
:::

::: {.cell .markdown id="3RjZ8LX86cFM"}
\#\#Using Reinforcement Learning to Buy or Sell on Stock Market
:::

::: {.cell .markdown id="iaipWLBF6jV2"}
\#\#\#Making Indicator for buy or sell signals
:::

::: {.cell .code execution_count="13" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="K_GcK5sg6qQM" outputId="fba8d9b8-edca-40cc-d532-b91d8c094037"}
``` {.python}
Endexler=Degisim_ortalaması_1Yil.index
Güvenli_liman=[]
Güvensiz_liman=[]
A=Degisim_ortalaması_1Yil.mean()
for i in range(6):
  if Degisim_ortalaması_1Yil[i]<=A:
    Güvensiz_liman.append(Endexler[i])
  else:
    Güvenli_liman.append(Endexler[i])

Güvenli_liman2=[]
Güvensiz_liman2=[]
A=Degisim_ortalaması_1Hafta.mean()
for i in range(6):
  if Degisim_ortalaması_1Hafta[i]<=A:
    Güvensiz_liman2.append(Endexler[i])
  else:
    Güvenli_liman2.append(Endexler[i])

Trend_Hisse = set(Güvenli_liman) & set(Güvenli_liman2)
print(Trend_Hisse)
```

::: {.output .stream .stdout}
    {'KozaAltin'}
:::
:::

::: {.cell .code execution_count="14" colab="{\"height\":458,\"base_uri\":\"https://localhost:8080/\"}" id="XwAhDdDY69hz" outputId="64459d5a-b235-46ee-8b4a-0e1de12fa999"}
``` {.python}
# Simple Moving Average 
def SMA(data, ndays): 
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name = 'SMA') 
    data = data.join(SMA) 
    return data

# Exponentially-weighted Moving Average 
def EWMA(data, ndays): 
    EMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays - 1).mean(), 
                 name = 'EWMA_' + str(ndays)) 
    data = data.join(EMA) 
    return data

# Retrieve the Goolge stock data from Yahoo finance
data = Akfen
close = data['Close']

# Compute the 50-day SMA
n = 50
SMA = SMA(data,n)
SMA = SMA.dropna()
SMA = SMA['SMA']

# Compute the 200-day EWMA
ew = 200
EWMA = EWMA(data,ew)
EWMA = EWMA.dropna()
EWMA = EWMA['EWMA_200']

# Plotting the Google stock Price Series chart and Moving Averages below
plt.figure(figsize=(10,7))

# Set the title and axis labels
plt.title('Moving Average')
plt.xlabel('Date')
plt.ylabel('Price')

# Plot close price and moving averages
plt.plot(data['Close'],lw=1, label='Close Price')
plt.plot(SMA,'g',lw=1, label='50-day SMA')
plt.plot(EWMA,'r', lw=1, label='200-day EMA')

# Add a legend to the axis
plt.legend()

plt.show()
```

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/dcde176741c16616cbf70fcc3250cee08ff34c66.png)
:::
:::

::: {.cell .code execution_count="15" colab="{\"height\":458,\"base_uri\":\"https://localhost:8080/\"}" id="Qme1gjKG7DEI" outputId="ab49b3c6-2ec0-47be-cf43-c59fc474f8bc"}
``` {.python}
# Compute the Bollinger Bands 
def BBANDS(data, window=n):
    MA = data.Close.rolling(window=n).mean()
    SD = data.Close.rolling(window=n).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data
 
# Retrieve the Goolge stock data from Yahoo finance
data = KozaAltin

# Compute the Bollinger Bands for Google using the 50-day Moving average
n = 50
BBANDS = BBANDS(data, n)

# Create the plot
# pd.concat([BBANDS.Close, BBANDS.UpperBB, BBANDS.LowerBB],axis=1).plot(figsize=(9,5),)

plt.figure(figsize=(10,7))

# Set the title and axis labels
plt.title('Bollinger Bands')
plt.xlabel('Date')
plt.ylabel('Price')

plt.plot(BBANDS.Close,lw=1, label='Close Price')
plt.plot(data['UpperBand'],'g',lw=1, label='Upper band')
plt.plot(data['MiddleBand'],'r',lw=1, label='Middle band')
plt.plot(data['LowerBand'],'g', lw=1, label='Lower band')

# Add a legend to the axis
plt.legend()

plt.show()
```

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/838eef1bfaf8e82304dad7ee049e25469ab6e14d.png)
:::
:::

::: {.cell .code execution_count="16" colab="{\"height\":530,\"base_uri\":\"https://localhost:8080/\"}" id="6lE4fEMC7G-p" outputId="70fcfeef-6736-4623-d015-5be2e71ba178"}
``` {.python}
# Calculate money flow index
def mfi(high, low, close, volume, n=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    mf_sign = np.where(typical_price > typical_price.shift(1), 1, -1)
    signed_mf = money_flow * mf_sign
    mf_avg_gain = signed_mf.rolling(n).apply(lambda x: (x > 0).sum(), raw=True)
    mf_avg_loss = signed_mf.rolling(n).apply(lambda x: (x < 0).sum(), raw=True)
    return 100 - (100 / (1 + (mf_avg_gain / abs(mf_avg_loss))))

# Retrieve the Apple Inc. data from Yahoo finance
data = yf.download("AAPL", start="2020-01-01", end="2022-04-30")

data['MFI'] = mfi(data['High'], data['Low'], data['Close'], data['Volume'], 14)

# Plotting the Price Series chart and the MFI below
fig = plt.figure(figsize=(10, 7))

# Define position of 1st subplot
ax = fig.add_subplot(2, 1, 1)

# Set the title and axis labels
plt.title('Apple Price Chart')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.plot(data['Close'], label='Close price')

# Add a legend to the axis
plt.legend()

# Define position of 2nd subplot
bx = fig.add_subplot(2, 1, 2)

# Set the title and axis labels
plt.title('Money flow index')
plt.xlabel('Date')
plt.ylabel('MFI values')

plt.plot(data['MFI'], 'm', label='MFI')

# Add a legend to the axis
plt.legend()

plt.tight_layout()
plt.show()
```

::: {.output .stream .stdout}
    [*********************100%***********************]  1 of 1 completed
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/b1ca6fde5a050f6e6bd8a65f8fc2f501be7aa516.png)
:::
:::

::: {.cell .code execution_count="45" id="aFdcJXXErOHC"}
``` {.python}
df= Vestel.copy()
count = int(np.ceil(len(df) * 0.1))
signals = pd.DataFrame(index=df.index)
signals['signal'] = 0.0
signals['trend'] = df['Close']
signals['RollingMax'] = (signals.trend.shift(1).rolling(count).max())
signals['RollingMin'] = (signals.trend.shift(1).rolling(count).min())
signals.loc[signals['RollingMax'] < signals.trend, 'signal'] = -1
signals.loc[signals['RollingMin'] > signals.trend, 'signal'] = 1
```
:::

::: {.cell .markdown id="DiKiAkRjkD6y"}
### Reinforcement İle Analiz {#reinforcement-i̇le-analiz}
:::

::: {.cell .code execution_count="46" colab="{\"height\":0,\"base_uri\":\"https://localhost:8080/\"}" id="JbzAv8FR1Fkl" outputId="a78348ed-c2e2-40d1-a874-e1fd9eef2cc8"}
``` {.python}
def buy_stock(
    real_movement,
    signal,
    initial_money = 10000,
    max_buy = 50000,
    max_sell = 50000,
):
    """
    real_movement = actual movement in the real world
    delay = how much interval you want to delay to change our decision from buy to sell, vice versa
    initial_state = 1 is buy, 0 is sell
    initial_money = 1000, ignore what kind of currency
    max_buy = max quantity for share to buy
    max_sell = max quantity for share to sell
    """
    starting_money = initial_money
    states_sell = []
    states_buy = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print(
                'day %d: total balances %f, not enough money to buy a unit price %f'
                % (i, initial_money, real_movement[i])
            )
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print(
                'day %d: buy %d units at price %f, total balance %f'
                % (i, buy_units, buy_units * real_movement[i], initial_money)
            )
            states_buy.append(0)
        return initial_money, current_inventory

    for i in range(real_movement.shape[0] - int(0.025 * len(df))):
        state = signal[i]
        if state == 1:
            initial_money, current_inventory = buy(
                i, initial_money, current_inventory
            )
            states_buy.append(i)
        elif state == -1:
            if current_inventory == 0:
                    print('day %d: cannot sell anything, inventory 0' % (i))
            else:
                if current_inventory > max_sell:
                    sell_units = max_sell
                else:
                    sell_units = current_inventory
                current_inventory -= sell_units
                total_sell = sell_units * real_movement[i]
                initial_money += total_sell
                try:
                    invest = (
                        (real_movement[i] - real_movement[states_buy[-1]])
                        / real_movement[states_buy[-1]]
                    ) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell %d units at price %f, investment %f %%, total balance %f,'
                    % (i, sell_units, total_sell, invest, initial_money)
                )
            states_sell.append(i)
            
    invest = ((initial_money - starting_money) / starting_money) * 100
    total_gains = initial_money - starting_money
    return states_buy, states_sell, total_gains, invest
states_buy, states_sell, total_gains, invest = buy_stock(df.Close, signals['signal'])
```

::: {.output .stream .stdout}
    day 263: cannot sell anything, inventory 0
    day 264: cannot sell anything, inventory 0
    day 265: cannot sell anything, inventory 0
    day 267: cannot sell anything, inventory 0
    day 270: cannot sell anything, inventory 0
    day 271: cannot sell anything, inventory 0
    day 273: cannot sell anything, inventory 0
    day 274: cannot sell anything, inventory 0
    day 275: cannot sell anything, inventory 0
    day 277: cannot sell anything, inventory 0
    day 287: cannot sell anything, inventory 0
    day 290: cannot sell anything, inventory 0
    day 307: cannot sell anything, inventory 0
    day 310: cannot sell anything, inventory 0
    day 311: cannot sell anything, inventory 0
    day 312: cannot sell anything, inventory 0
    day 332: buy 1088 units at price 9998.719543, total balance 1.280457
    day 334: total balances 1.280457, not enough money to buy a unit price 8.910000
    day 415, sell 1088 units at price 19224.960083, investment 98.316503 %, total balance 19226.240540,
    day 416: cannot sell anything, inventory 0
    day 445: cannot sell anything, inventory 0
    day 446: cannot sell anything, inventory 0
    day 481: cannot sell anything, inventory 0
    day 494: cannot sell anything, inventory 0
    day 495: cannot sell anything, inventory 0
    day 496: cannot sell anything, inventory 0
    day 509: cannot sell anything, inventory 0
    day 517: cannot sell anything, inventory 0
    day 518: cannot sell anything, inventory 0
    day 519: cannot sell anything, inventory 0
    day 537: cannot sell anything, inventory 0
    day 538: cannot sell anything, inventory 0
    day 540: cannot sell anything, inventory 0
    day 541: cannot sell anything, inventory 0
    day 542: cannot sell anything, inventory 0
    day 550: cannot sell anything, inventory 0
    day 551: cannot sell anything, inventory 0
    day 552: cannot sell anything, inventory 0
    day 553: cannot sell anything, inventory 0
    day 556: cannot sell anything, inventory 0
    day 557: cannot sell anything, inventory 0
    day 558: cannot sell anything, inventory 0
    day 559: cannot sell anything, inventory 0
    day 568: cannot sell anything, inventory 0
    day 578: cannot sell anything, inventory 0
    day 581: cannot sell anything, inventory 0
    day 585: cannot sell anything, inventory 0
    day 588: cannot sell anything, inventory 0
    day 590: cannot sell anything, inventory 0
    day 591: cannot sell anything, inventory 0
    day 592: cannot sell anything, inventory 0
    day 594: cannot sell anything, inventory 0
    day 595: cannot sell anything, inventory 0
    day 682: buy 668 units at price 19225.040459, total balance 1.200081
    day 693: total balances 1.200081, not enough money to buy a unit price 28.480000
    day 707: total balances 1.200081, not enough money to buy a unit price 28.440001
    day 708: total balances 1.200081, not enough money to buy a unit price 27.900000
    day 710: total balances 1.200081, not enough money to buy a unit price 27.540001
    day 717: total balances 1.200081, not enough money to buy a unit price 26.299999
    day 718: total balances 1.200081, not enough money to buy a unit price 25.760000
    day 719: total balances 1.200081, not enough money to buy a unit price 25.360001
    day 720: total balances 1.200081, not enough money to buy a unit price 24.860001
    day 721: total balances 1.200081, not enough money to buy a unit price 24.660000
    day 726: total balances 1.200081, not enough money to buy a unit price 24.160000
    day 727: total balances 1.200081, not enough money to buy a unit price 23.879999
    day 728: total balances 1.200081, not enough money to buy a unit price 23.379999
    day 819: total balances 1.200081, not enough money to buy a unit price 23.200001
    day 820: total balances 1.200081, not enough money to buy a unit price 23.100000
    day 821: total balances 1.200081, not enough money to buy a unit price 20.799999
    day 949, sell 668 units at price 21416.080917, investment 54.134628 %, total balance 21417.280998,
    day 950: cannot sell anything, inventory 0
    day 952: cannot sell anything, inventory 0
    day 964: cannot sell anything, inventory 0
    day 965: cannot sell anything, inventory 0
    day 967: cannot sell anything, inventory 0
    day 968: cannot sell anything, inventory 0
    day 969: cannot sell anything, inventory 0
    day 970: cannot sell anything, inventory 0
    day 973: cannot sell anything, inventory 0
    day 974: cannot sell anything, inventory 0
    day 982: cannot sell anything, inventory 0
    day 985: cannot sell anything, inventory 0
    day 986: cannot sell anything, inventory 0
    day 988: cannot sell anything, inventory 0
    day 995: cannot sell anything, inventory 0
:::
:::

::: {.cell .code execution_count="47" colab="{\"height\":336,\"base_uri\":\"https://localhost:8080/\"}" id="-tV4bZ18Ps_N" outputId="2e398b36-6fca-4664-99e1-d5fb5dd05c2a"}
``` {.python}
close = df['Close']
fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
plt.show()
```

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/af2ef834ac6398cfa495375727d24e75f8ecb126.png)
:::
:::

::: {.cell .markdown id="UQiBaVKw2q-3"}
\#İndikatörlerle Uygulamalar
:::

::: {.cell .code execution_count="23" colab="{\"height\":357,\"base_uri\":\"https://localhost:8080/\"}" id="JLofBENv6gOC" outputId="65b7c0ed-d19e-420f-e731-fdcf4868ce7f"}
``` {.python}
import numpy as np
import pandas as pd

# Load the stock data
data = KozaAltin

# Normalize the data
data = (data - data.mean()) / data.std()

# Split the data into training and test sets
train_data = data[:len(data)]


# Define the learning rate
learning_rate = 0.1

# Define the discount factor
discount_factor = 0.95

# Define the exploration rate
exploration_rate = 0.1

# Define the number of actions
num_actions = 3

# Define the Q-table
q_table = np.zeros((len(train_data), num_actions))

# Define the rewards
rewards = []

# Define lists to store the buying and selling points
buying_points = []
selling_points = []

# Train the agent
for i in range(len(train_data)-1):
    # Choose an action using the exploration rate
    if np.random.uniform() < exploration_rate:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q_table[i])
        
    # Take the action and observe the reward
    if action == 0:
        # Buy
        reward = train_data.iloc[i+1]['Close'] - train_data.iloc[i]['Close']
        buying_points.append(i)
    elif action == 1:
        # Sell
        reward = train_data.iloc[i]['Close'] - train_data.iloc[i+1]['Close']
        selling_points.append(i)
    else:
        # Hold
        reward = 0
        
    # Update the Q-table
    q_table[i+1] = q_table[i] + learning_rate * (reward + discount_factor * np.max(q_table[i+1]) - q_table[i])
    
    # Add the reward to the list of rewards
    rewards.append(reward)
    

# Print the final rewards
print(rewards)

close = KozaAltin['Close']
fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = buying_points)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = selling_points)
plt.legend()
plt.show()
```

::: {.output .stream .stdout}
    [0.024320754681997014, 0.01158135382875114, -0.004632574665893174, -0.038218313007713056, 0.04748340711551158, -0.018530132991609216, -0.030111376372384813, -0.026637014402949633, 0.06253906768971007, -0.04400898992208868, 0.01273945607723359, 0.0, -0.009265038883810695, -0.006948779162858076, -0.015055715798186209, -0.040534572728665896, 0.001158102248482562, 0.037060210759230605, 0.01853007776762139, -0.030111376372384813, 0.008106936635328243, 0.0, -0.022004494961044396, 0.009265038883810695, -0.00115810224848234, 0.019688235240091667, 0.006948779162857965, 0.026637014402949744, -0.005790676914375625, 0.011581298604763424, -0.003474361969435069, 0.02895327412390236, -0.009265094107798522, -0.02895327412390236, -0.012739400853245875, -0.010423196356280973, -0.011581298604763424, -0.008106881411340527, -0.009265094107798522, 0.039376470480183334, -0.01505571579818632, -0.008106881411340416, -0.02895327412390236, -0.04864150936399414, -0.02547891215446718, -0.03706015553524278, -0.022004494961044507, 0, -0.011581298604763202, 0, -0.003474361969435069, 0.010423141132293035, -0.01853007776762139, 0.009265038883810695, 0, -0.053274028805899265, 0.019688235240091556, 0.0034744171934228962, 0.002316259720952729, 0.019688235240091778, -0.01158135382875125, 0.02779517187541991, -0.010423196356280862, 0.009265038883810695, 0.02779517187541991, -0.01737197551913905, 0.001158157472470167, -0.002316314944940334, -0.03821831300771317, -0.008106881411340527, -0.002316259720952507, -0.01158135382875125, 0.016213873270656487, 0.003474361969435069, 0.002316259720952729, 0.05095776908494676, -0.01273945607723359, -0.013897558325716153, -0.00115810224848234, -0.019688235240091778, -0.010423196356280862, 0.006948779162857965, -0.024320754681997014, 0.0034744171934228962, 0.002316259720952729, 0.024320754681997014, 0, -0.009265038883810695, 0.019688235240091778, 0.010423196356280862, -0.01389761354970398, 0.002316259720952729, 0.0, 0.0, -0.006948779162857965, 0.0, 0.005790676914375625, -0.0011581574724703891, -0.005790621690387798, 0.002316259720952729, -0.008106936635328132, 0.0, -0.003474361969435291, -0.016213873270656487, 0.001158157472470167, -0.010423196356280862, 0.023162652433514674, 0.0, -0.010423196356281084, -0.002316259720952729, -0.002316259720952507, 0.005790621690387798, 0.0069488343868457925, -0.010423196356281084, 0.03242763609333754, -0.022004439737056458, 0.03590205328676044, 0.03474389581429005, -0.06022280796875745, 0.009265094107798522, 0.03358579356580771, 0.003474361969435291, -0.025478856930479576, -0.020846392712561945, 0.01505571579818632, -0.019688235240091556, 0.024320754681997014, -0.01273945607723359, 0.03242769131732515, -0.010423196356280862, 0.024320754681997014, 0.0, 0.0, 0.0, 0.03011143159637264, 0.01621381804666866, -0.0011581574724702781, 0.01853007776762139, 0.002316259720952729, 0.005790676914375625, 0.009265038883810695, 0.012739456077233702, -0.04169273020113595, -0.004632519441905347, -0.006948779162858076, 0.020846337488574118, -0.020846337488574118, 0.0011581574724702781, -0.002316259720952729, 0, 0.009265038883810806, 0.026637014402949633, -0.028953274123902473, -0.020846337488574118, -0.026637014402949744, -0.00115810224848234, -0.03242769131732537, 0.0347439510382781, -0.008106936635328355, 0.002316259720952729, 0.0, -0.009265038883810695, 0.03474389581429005, 0.01505571579818632, 0.010423196356281084, -0.012739456077233813, 0.020846392712561945, 0.001158102248482562, -0.01621381804666866, -0.003474361969435069, 0.026637014402949633, 0.008106881411340527, -0.00347436196943518, 0.011581298604763424, -0.01737197551913905, 0.040534572728665785, 0.024320754681997125, -0.01621381804666877, 0, 0.06369716993819252, 0.0, 0.0, 0.0, 0.010423196356280973, 0.05559034375083993, -0.019688235240091667, 0.04748335189152375, -0.1297106548213256, 0.013897558325716153, 0, 0.01853007776762139, 0.002316204496964902, 0.017372030743126765, -0.035902108510748154, 0.0, -0.01853007776762139, 0.01737192029515111, -0.013897558325716042, -0.030111376372384813, 0.008106936635328243, -0.016213873270656487, 0.002316259720952618, 0.025478912154467293, -0.01621381804666877, 0, 0.01621381804666877, 0.0069488343868457925, 0.017371920295151222, -0.03706015553524289, 0.031269478620867264, 0.009265038883810695, 0.025478912154467293, 0.03126958906884292, 0.023162597209526736, -0.005790676914375625, 0.025478912154467293, 0.02200443973705657, -0.01505571579818632, 0.01505571579818632, 0.020846392712561834, -0.02895327412390236, 0.026636959178961916, -0.03474384059030233, 0.02432064423402136, -0.02432064423402136, -0.04979966683646431, 0.023162597209526736, -0.07875294096036667, -0.005790566466399971, -0.00810699185931607, 0.0, 0.02663706962693746, 0.0185300777676215, 0.011581243380775597, 0, -0.022004439737056458, -0.009265038883810806, -0.031269478620867264, -0.022004439737056458, 0.006948723938870138, -0.013897558325716042, -0.009265038883810695, 0.024320754681997014, -0.012739400853245764, 0.026636959178961805, 0.023162597209526847, 0.04285094289759406, 0.010423085908305318, 0.02663706962693746, 0.008106881411340527, -0.013897558325716042, 0.02663706962693746, -0.027795227099407738, -0.005790566466399971, -0.020846392712561945, -0.012739511301221418, -0.004632519441905347, 0.009265149331786349, -0.009265149331786349, 0, -0.001158157472470167, 0.008106881411340416, 0.020846392712561945, -0.013897558325716042, -0.019688235240091667, -0.0011581574724702781, 0.0034744724174107233, 0.008106881411340416, -0.02663706962693746, -0.009265038883810695, 0.009265038883810695, -0.02200443973705657, 0.02200443973705657, 0.07643673646340177, -0.013897668773691696, 0.04632530486702924, 0.019688235240091667, -0.027795116651432195, -0.022004550185032112, -0.013897558325716042, 0.0069488343868457925, -0.07759467303992085, -0.019688124792116013, 0.01737192029515111, -0.002316314944940445, 0, 0.016213873270656487, -0.02895327412390236, 0.027795116651432195, -0.002316204496964902, -0.009265038883810695, -0.019688235240091667, -0.009265038883810695, 0.023162597209526736, 0.005790676914375625, 0.06485527218667497, -0.048641398916018375, -0.00810699185931607, -0.01505571579818632, 0.0069488343868457925, -0.008106881411340416, -0.0069488343868459035, 0.0011581574724702781, 0.003474361969435069, 0.0185300777676215, 0.024320754681997014, -0.011581243380775597, 0.019688235240091667, -0.039376470480183334, 0.022004439737056458, 0.009265038883810806, 0.038218313007713056, 0.010423196356280973, 0.023162597209526736, 0.0011581574724702781, 0.016213873270656542, -0.04979966683646425, 0.148240732588947, 0, -0.011581243380775597, -0.006948834386845848, -0.060222752744769625, -0.06485538263465057, 0.019688235240091667, 0.07875294096036667, -0.06253906768971007, -0.016213873270656598, -0.003474361969435069, -0.12623629285189053, -0.09612486125551789, 0.0, -0.07991098798486129, 0.08570166489923692, -0.11928745846504463, -0.04979966683646431, -0.024320754681997014, 0.010423196356280973, 0.009265038883810695, 0.0, 0.09033418434114227, 0.05559034375083993, 0.08685982237170708, -0.09149234181361243, -0.0069488343868459035, 0.016213873270656598, -0.027795227099407738, 0.020846392712561834, 0.008106881411340527, 0.08338546040227202, 0.01158135382875114, 0.05327402880589949, 0.06948779162858032, -0.009265038883810695, 0.016213873270656487, 0.0706459491010506, -0.06253906768971007, 0.05095782430893453, -0.0185301882155971, -0.003474361969435069, -0.025478912154467293, 0.04632530486702918, 0.017371920295151166, 0.08570166489923692, -0.06948779162858032, -0.01273940085324582, -0.03011143159637264, 0.008106881411340472, -0.039376470480183334, 0.0567483907753345, -0.025478912154467293, -0.08685982237170714, -0.024320754681997014, 0.09380865675855299, 0.04516714739455896, -0.006948834386845848, 0.03358579356580771, 0, -0.01158135382875125, 0.028953274123902417, 0.0023163149449405007, -0.0034743619694351247, 0.019688235240091667, -0.054432186278369654, -0.018530077767621445, 0.026636959178961916, -0.017371920295151166, -0.0023163149449405007, 0.006948834386845848, -0.03011143159637264, 0.033585793565807764, 0.015055715798186264, 0.02200443973705657, -0.00926503888381075, 0.048641509363994084, -0.01273940085324582, 0.0, 0.003474361969435069, 0.01158135382875125, 0.07180410657352082, 0.0023163149449405007, 0.00810699185931607, -0.03706026598321849, -0.020846282264586236, 0.04632519441905353, 0.0023163149449405007, -0.0034744724174107233, -0.004632519441905347, -0.02084628226458629, 0.05790654824780475, 0.0011580470244945962, -0.008106991859316098, -0.07180410657352079, -0.023162707657502446, 0.04285094289759411, -0.0046326298898810014, 0.04864150936399403, 0.08338546040227204, 0, -0.04400898992208868, 0.04400898992208868, 0.016213762822680944, -0.0069487239388702216, 0.045167036946583305, -0.07296226404599107, 0.02895327412390239, -0.02779511665143214, -0.05559023330286428, 0.08454350742676667, -0.07759478348789645, -0.018530077767621417, -0.038218313007713084, 0.010423196356280973, 0.00926503888381075, -0.04285083244961846, -0.0034743619694351247, 0.023162597209526792, 0.003474361969435069, 0.06485538263465063, -0.03126958906884286, -0.047483351891523806, 0.0011581574724702226, -0.03590210851074821, 0.0011581574724702781, 0.05790654824780472, -0.02779511665143214, -0.01853007776762139, 0.01273940085324582, -0.017371920295151166, 0.010423196356280917, -0.00579067691437557, 0.013897558325716042, 0.016213873270656598, 0.022004439737056514, 0, -0.006948723938870194, 0.004632519441905347, 0.010423196356280973, -0.005790676914375625, -0.010423196356280973, 0.020846392712561945, -0.06253906768971013, 0, -0.005790676914375625, 0.0034743619694351247, 0.008106881411340472, -0.016213762822680944, 0.002316204496964902, 0.004632519441905347, -0.041692785425123835, 0.0023162044969648465, 0.06253906768971013, -0.019688235240091667, 0.022004550185032168, -0.03011143159637264, 0.03590210851074821, 0.010423085908305346, -0.03706015553524286, 0.01273940085324582, -0.02779511665143214, -0.0023162044969648465, 0.032427636093337486, 0.006948834386845848, 0.01968812479211604, -0.044008879474113055, -0.01505571579818632, -0.03821831300771311, 0.005790566466399971, -0.0428508324496184, -0.047483351891523806, -0.008106881411340472, 0.03474395103827799, -0.003474472417410779, -0.05327402880589943, 0.03126958906884286, -0.04632530486702918, -0.017371920295151166, -0.01505571579818632, -0.023162597209526792, 0.032427636093337486, 0.005790676914375625, -0.013897558325716097, 0.00926503888381075, -0.008106881411340472, 0.14592441764400652, 0, 0.013897668773691696, -0.027795227099407738, 0.03242774654131311, -0.055590343750839905, -0.054432186278369654, 0.02895327412390236, 0.025478912154467293, -0.0023162044969648465, -0.0011581574724702781, 0.01273940085324582, 0.011581353828751195, 0.047483351891523834, 0.008106881411340472, -0.011581353828751195, -0.032427636093337486, 0.016213873270656542, 0.006948834386845848, 0.018530077767621417, -0.016213873270656542, 0.01273940085324582, -0.009265038883810722, 0.025478912154467265, -0.1656127633320738, 0.006948723938870218, -0.02547880170649164, -0.004632519441905354, 0.008106881411340463, -0.019688235240091667, 0.12507824582739585, -0.07643673646340181, -0.017372030743126807, 0.020846392712561914, 0, 0.10075749114539884, 0.005790566466399971, -0.09380854631057736, -0.028953274123902376, 0.01737192029515118, -0.003474472417410737, -0.005790787362351238, -0.016213762822680923, 0.0023163149449404938, -0.03474406148625361, 0.026636959178961882, -0.017371920295151173, -0.09149223136563686, 0.030111321148396996, -0.009265038883810712, 0, -0.04748346233949943, 0.03590210851074823, -0.03706015553524285, 0.046325194419053556, 0.035901998062772604, 0.008106881411340458, 0.05790654824780475, 0.11928756891302025, 0.03821831300771311, 0.1656127633320738, 0.04053462795265361, -0.08222741337777745, 0.12623629285189047, -0.1042317426668583, -0.08454350742676664, -0.11002253002920959, -0.01737192029515114, -0.08106925590530717, 0.009265038883810695, 0, -0.1158130964956095, -0.047483351891523806, -0.03126958906884287, 0.003474472417410744, 0.004632408993929721, -0.026636959178961885, 0.02895327412390238, -0.04516714739455894, -0.009265038883810712, -0.03126947862086724, 0.020846282264586284, -0.060222752744769625, -0.010423196356280959, 0, -0.1679290782770143, 0.003474472417410779, 0.02084628226458629, 0.060222863192745224, 0.03821831300771311, 0.06253906768971013, -0.02895327412390236, 0.017371920295151166, 0.047483351891523806, 0.07412042151846131, 0.03242763609333749, -0.026636959178961882, -0.01621387327065657, -0.018530077767621417, -0.019688235240091667, -0.06485527218667497, -0.011581243380775569, 0.0567483907753345, -0.02895327412390236, 0.011581353828751195, -0.05211587133342915, -0.04979966683646428, -0.0671715871316155, 0.05790654824780478, 0.017371920295151166, 0.013897668773691696, -0.04864150936399403, 0.023162597209526792, 0.025478912154467237, 0.02895327412390239, -0.0011581574724702504, 0.005790676914375598, 0.05790654824780475, 0.07527846854295595, 0.022004439737056534, 0.017372030743126803, -0.025478912154467268, 0.08917613731664763, 0.09959911277697732, 0.038218533903664365, 0.004632408993929721, 0.019688235240091667, -0.09033429478911788, -0.01505560535021068, -0.040534627952653585, -0.06138091021723987, 0, -0.0011581574724702469, 0.0011581574724702469, -0.031269589068842876, 0.1204457263854905, 0.0011581574724702504, -0.04979966683646429, 0.0034742515214594843, 0.0046326298898809876, -0.028953274123902382, -0.04516714739455894, -0.002316204496964862, -0.02663706962693752, -0.07296226404599107, -0.045167036946583305, -0.01621387327065657, 0.020846392712561918, 0.05211587133342915, 0.01737192029515118, -0.02663695917896189, -0.045167147394558946, 0.004632519441905375, -0.04053451750467796, 0.01389755832571607, -0.053274028805899404, -0.01505571579818632, 0.0011581574724702504, 0.0, -0.006948834386845848, 0.003474361969435097, -0.03358579356580771, -0.006948723938870249, 0.04864150936399406, -0.024320754681997042, -0.015055715798186264, -0.020846392712561945, 0.009265038883810695, 0.0023163149449405007, 0.053274028805899404, -0.006948834386845848, 0, 0.019688235240091667, 0.0023163149449405007, 0.039376470480183334, -0.030111431596372612, -0.039376470480183334, 0.002316314944940473, -0.02895327412390236, 0.019688235240091667, -0.0023163149449405007, 0.07180410657352082, 0.03011143159637264, -0.02895327412390239, 0.06601342965914522, 0.09496681423102322, 0.015055605350210683, -0.016213762822680926, -0.0034744724174107407, 0.03474395103827798, 0.011581353828751206, 0.0023162044969648604, 0.03474406148625361, 0.017371920295151173, -0.024320865129972648, 0.005790787362351231, 0, -0.03358579356580774, 0.026636959178961885, -0.020846282264586284, 0.07643651556745054, -0.009265038883810708, -0.061380799769264235, -0.08106925590530717, -0.05327402880589942, -0.030111321148396986, -0.01505571579818632, 0.018530077767621445, 0.03358579356580771, -0.05211587133342915, 0.04285083244961846, 0.03358579356580774, -0.03011143159637264, 0.01389755832571607, -0.0023162044969648465, 0.004632519441905347, -0.008106881411340472, -0.004632629889880974, 0.04632530486702918, 0.009265038883810708, 0.010423196356280959, 0.02316259720952677, -0.003474361969435104, 0.022004439737056528, -0.02663695917896189, 0.039376470480183334, 0.0416926749771482, -0.023162597209526778, 0.024320754681997025, 0.04053451750467795, 0.03474406148625361, 0.01621376282268093, 0.008106881411340458, 0.011581353828751195, -0.026636959178961875, -0.08106925590530717, -0.030111431596372626, 0.07991109843283692, 0.005790566466399971, 0.042850942897594085, -0.020846392712561918, 0.023162486761551138, 0, 0.006948723938870194, 0.09265038883810708, 0.019688235240091667, -0.005790566466399971, 0.023162707657502446, 0.02200432928908086, -0.02547880170649164, 0.09380854631057739, -0.04400887947411308, -0.005790787362351224, 0.02432086512997267, -0.017371920295151222, 0.12971054437335, 0.02316270765750239, -0.03011143159637264, 0.14824084303692264, 0.020846171816610637, 0.04632541531500478, -0.01853007776762139, -0.025479022602442836, 0.03706015553524278, 0.10886437255673931, 0.002316314944940445, -0.006948723938870138, 0.1227618204344797, -0.20614739128472737, -0.10191542772191786, -0.24436570429244048, -0.20730532786124636, -0.04864150936399404, 0.10770599418831779, 0.02663718007491314, 0, 0.016213762822680944, -0.019688235240091667, 0.032427746541313085, 0, 0.0521157608854535, 0.1297107652693012, 0.027795116651432195, -0.020846392712561945, 0.027795116651432084, -0.05559023330286428, 0.1366594892081715, -0.08106925590530722, 0.06485549308262628, 0.19919844644990592, -0.1366594892081714, 0.08338557085024756, -0.07875294096036667, 0.10191542772191786, -0.12507813537942025, 0.12971076526930125, 0.0, -0.01853007776762139, 0.013897447877740499, 0.0, 0.002316314944940445, -0.06485527218667497, -0.12971076526930125, -0.009265038883810695, 0.06717158713161553, -0.006948723938870249, 0.06253895724173453, -0.03242752564536189, -0.020846392712561945, 0.032427746541313085, 0.006948944834821447, 0.009265038883810695, 0.020846171816610637, 0.03011143159637264, -0.02547880170649164, -0.03474384059030233, -0.03011143159637264, -0.2478401767098512, 0.09959933367292856, -0.06253895724173442, 0.09496670378304761, -0.05443229672634531, -0.23046803551874873, 0.0231627076575025, 0.12044572638549045, 0.2686863485264619, -0.009265038883810695, 0.009265038883810695, -0.11349678155066911, -0.06485549308262617, 0.016213983718632252, 0.06253895724173442, 0.06253917813768584, 0.04864150936399403, 0.03011121070042133, 0.04632541531500478, -0.03474406148625353, -0.03011143159637264, -0.04169256452917258, 0.004632408993929582, -0.05559023330286417, 0.07875294096036667, -0.03011143159637264, 0.05327391835792383, 0.2316261929912189, 0.009265038883810695, 0.027795116651432306, -0.06253895724173453, 0.01853007776762139, 0, 0.03706015553524278, 0.3335418416090883, -0.04169278542512389, 0, 0.21541243016853762, -0.08801820074012845, 0.06022286319274528, -0.039376470480183334, -0.2362586019851487, 0.00694872393887036, -0.08570166489923703, 0.057906548247804945, -0.06948790207655597, 0.1459245280919821, -0.14360821314704153, 0.15518956697579278, 0.09033407389316661, 0.00926525977976178, -0.09033451568506878, -0.13897536236120955, 0.11812941144055, 0, -0.07412031107048556, -0.12971076526930125, -0.07180421702149631, -0.16677092080454403, 0.07412053196643709, -0.006948944834821669, 0.06948790207655597, -0.14824062214097133, 0.04169256452917258, 0.020846392712561945, 0.07412031107048578, 0.07412053196643686, -0.050957824308934585, 0.0, 0.039376470480183334, -0.3312255266641475, 0.06022286319274506, -0.17835205373734397, -0.06022286319274506, 0.0625389572417343, 0.013897668773691807, -0.12276182043447981, 0.24783995581389995, 0.020846392712561945, 0.009265038883810695, -0.07412053196643686, -0.06485527218667508, 0.05327413925387514, -0.07643662601542633, -0.07643662601542611, -0.1598221968656739, 0.01853007776762161, 0.2362588228811, 0.05095760341298328, -0.1505569370859119, -0.0231627076575025, -0.04864150936399403, -0.009265038883810695, -0.09265038883810717, -0.07875294096036667, 0.09033429478911792, 0.009265038883810695, 0.039376470480183334, -0.03011143159637264, 0.046325194419053695, 0.07180421702149631, 0.09728301872798806, -0.055590454198815475, 0.08801797984417736, 0.04400910037006445, 0.07180399612554522, -0.03011143159637264, -0.011581132932799942, 0.12507813537942014, -0.05327413925387514, 0.08570166489923703, 0.006948723938870138, -0.345122974541888, -0.12507813537942014, -0.00694872393887036, -0.06948790207655597, -0.025478801706491527, 0.06022286319274528, 0.08801775894822605, -0.034743840590302444, 0.06022286319274528, -0.00463262988988089, -0.05095760341298328, -0.07180399612554522, -0.04169278542512389, 0.0, 0.09496692467899881, -0.06948790207655597, 0.07875294096036667, 0.06022264229679397, -0.032427525645361666, 0.226993563101338, 0.03011143159637264, 0.04632519441905347, 0, -0.3034304100127154, 0, 0, 0.002316094048989248, -0.18993340756609522, 0.016213762822680833, 0.21541243016853828, -0.07875294096036667, 0.020846392712561945, -0.06253917813768584, -0.11581309649560945, -0.06948768118060489, -0.09728301872798806, -0.00463262988988089, 0.2223611541074082, -0.006948723938870138, -0.025478801706491527, 0.18298468362722486, -0.08570188579518812, -0.03474406148625353, 0.08106903500935592, -0.025478801706491527, -0.08338534995429647, 0.19456603745597612, 0.31501132204956406, -0.10191520682596655, -0.07412053196643686, 0.00694872393887036, 0.06022286319274528, -0.03242752564536189, -0.08801820074012845, -0.004632408993929582, -0.004632408993929582, 0, 0.13434317426323084, 0.027795116651432084, 0.09033451568506923, 0.07412031107048556, 0.002316314944940334, 0.29879755922688345, -0.07180399612554522, 0.12971054437335017, 0.081069255905307, -0.108864151660788, 0.16908701485353328, -0.09033407389316661, 0.1320268593182905, 0.0857014440032855, -0.16908701485353328, 0.14592430719603078, 0.22931031983818118, -0.002316314944940334, 0.47020088902535706, 0.013897889669642893, -0.05327436014982645, 0.06485549308262639, 0.07643662601542589, 0.5257915641201238, 0.04632497352310283, -0.20846348533371684, -0.12739422942840983, -0.19224972251103534, 0.528107437273162, 0.15055737887781362, 0.5964371818772483]
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/d40876d80a618f6490c29f350facbf08d2672925.png)
:::
:::

::: {.cell .code execution_count="24" colab="{\"height\":375,\"base_uri\":\"https://localhost:8080/\"}" id="n8vAVTTjMQY8" outputId="3ef871df-d4c3-4642-adf3-916d455685ed"}
``` {.python}
import numpy as np
import pandas as pd

# Load the stock data
data = KozaAltin

# Normalize the data
data = (data - data.mean()) / data.std()

# Split the data into training and test sets
train_data = data[:len(data)]

# Define the learning rate
learning_rate = 0.1

# Define the discount factor
discount_factor = 0.95

# Define the exploration rate
exploration_rate = 0.1

# Define the number of actions
num_actions = 3

# Define the Q-table
q_table = np.zeros((len(train_data), num_actions))

# Define the rewards
rewards = []

# Define lists to store the buying and selling points
buying_points = []
selling_points = []

# Define the cooldown period (in days)
cooldown_period = 20

# Initialize the cooldown counter
cooldown_counter = 0

# Train the agent
for i in range(len(train_data)-1):
    # Check if the agent is on cooldown
    if cooldown_counter > 0:
        # Decrement the cooldown counter
        cooldown_counter -= 1
    else:
        # Choose an action using the exploration rate
        if np.random.uniform() < exploration_rate:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[i])
        
        # Take the action and observe the reward
        if action == 0:
            # Buy
            reward = train_data.iloc[i+1]['Close'] - train_data.iloc[i]['Close']
            buying_points.append(i)
            # Set the cooldown counter
            cooldown_counter = cooldown_period
        elif action == 1:
            # Sell
            reward = train_data.iloc[i]['Close'] - train_data.iloc[i+1]['Close']
            selling_points.append(i)
            # Set the cooldown counter
            cooldown_counter = cooldown_period
        else:
            # Hold
            reward = 0
        
    # Update the Q-table
    q_table[i+1] = q_table[i] + learning_rate * (reward + discount_factor * np.max(q_table[i+1]) - q_table[i])
    
    # Add the reward to the list of rewards
    rewards.append(reward)
    

# Print the final rewards
print(rewards)

close = KozaAltin['Close']
fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = buying_points)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = selling_points)
plt.legend
```

::: {.output .stream .stdout}
    [0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, -0.010423196356280862, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.0011581574724702781, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, -0.00347436196943518, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, 0.02663706962693746, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, -0.027795227099407738, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.07643673646340177, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.016213873270656542, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, 0.010423196356280973, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, -0.0185301882155971, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, 0.028953274123902417, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, -0.0023163149449405007, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, 0.016213762822680944, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.03590210851074821, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.044008879474113055, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, -0.009265038883810722, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, 0.0023163149449404938, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1042317426668583, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.1679290782770143, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.04979966683646428, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, -0.09033429478911788, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.05211587133342915, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.053274028805899404, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.011581353828751206, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.03358579356580774, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02432086512997267, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, 0.12971076526930125, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.14824062214097133, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.04400910037006445, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, -0.06948768118060489, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.004632408993929582, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893, 0.013897889669642893]
:::

::: {.output .execute_result execution_count="24"}
    <function matplotlib.pyplot.legend(*args, **kwargs)>
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/4a6ed34434f04af3478b3d575804450c98ae0624.png)
:::
:::

::: {.cell .code execution_count="25" colab="{\"height\":392,\"base_uri\":\"https://localhost:8080/\"}" id="s3Xskbj3NDXY" outputId="18ad3ea1-7f7d-484c-beb9-f3c148f03f93"}
``` {.python}
import numpy as np
import pandas as pd

# Load the stock data
data = KozaAltin

# Normalize the data
data = (data - data.mean()) / data.std()

# Split the data into training and test sets
train_data = data[:len(data)]

# Define the learning rate
learning_rate = 0.1

# Define the discount factor
discount_factor = 0.95

# Define the exploration rate
exploration_rate = 0.1

# Define the number of actions
num_actions = 3

# Define the Q-table
q_table = np.zeros((len(train_data), num_actions))

# Define the rewards
rewards = []

# Define lists to store the buying and selling points
buying_points = []
selling_points = []

# Define the cooldown period (in days)
cooldown_period = 5

# Initialize the cooldown counter
cooldown_counter = 0

# Define the starting amount of money
starting_money = 1000

# Initialize the current amount of money
money = starting_money

# Train the agent
for i in range(len(train_data)-1):
    # Check if the agent is on cooldown
    if cooldown_counter > 0:
        # Decrement the cooldown counter
        cooldown_counter -= 1
    else:
        # Choose an action using the exploration rate
        if np.random.uniform() < exploration_rate:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(q_table[i])
        
        # Take the action and observe the reward
        if action == 0:
            # Buy
            reward = train_data.iloc[i+1]['Close'] - train_data.iloc[i]['Close']
            buying_points.append(i)
            # Set the cooldown counter
            cooldown_counter = cooldown_period
            # Update the amount of money
            money -= train_data.iloc[i]['Close']
        elif action == 1:
            # Sell
            reward = train_data.iloc[i]['Close'] - train_data.iloc[i+1]['Close']
            selling_points.append(i)
            # Set the cooldown counter
            cooldown_counter = cooldown_period
            # Update the amount of money
            money += train_data.iloc[i]['Close']
        else:
            # Hold
            reward = 0
        
    # Update the Q-table
    q_table[i+1] = q_table[i] + learning_rate * (reward + discount_factor * np.max(q_table[i+1]) - q_table[i])
    
    # Add the reward to the list of rewards
    rewards.append(reward)
    

# Print the final rewards
print(rewards)

# Print the final amount of money
print("Final amount of money:", money)

close = KozaAltin['Close']
fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = buying_points)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = selling_points)
plt.legend
```

::: {.output .stream .stdout}
    [0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, -0.030111376372384813, -0.030111376372384813, -0.030111376372384813, -0.030111376372384813, -0.030111376372384813, -0.030111376372384813, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, 0.01853007776762139, 0.01853007776762139, 0.01853007776762139, 0.01853007776762139, 0.01853007776762139, 0.01853007776762139, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.003474361969435069, -0.003474361969435069, -0.003474361969435069, -0.003474361969435069, -0.003474361969435069, -0.003474361969435069, 0.011581298604763424, 0.011581298604763424, 0.011581298604763424, 0.011581298604763424, 0.011581298604763424, 0.011581298604763424, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.02895327412390236, -0.011581298604763202, -0.011581298604763202, -0.011581298604763202, -0.011581298604763202, -0.011581298604763202, -0.011581298604763202, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01158135382875125, -0.01158135382875125, -0.01158135382875125, -0.01158135382875125, -0.01158135382875125, -0.01158135382875125, 0.001158157472470167, 0.001158157472470167, 0.001158157472470167, 0.001158157472470167, 0.001158157472470167, 0.001158157472470167, 0.016213873270656487, 0.016213873270656487, 0.016213873270656487, 0.016213873270656487, 0.016213873270656487, 0.016213873270656487, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.00115810224848234, -0.002316259720952729, -0.002316259720952729, -0.002316259720952729, -0.002316259720952729, -0.002316259720952729, -0.002316259720952729, -0.01389761354970398, -0.01389761354970398, -0.01389761354970398, -0.01389761354970398, -0.01389761354970398, -0.01389761354970398, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, 0.005790676914375625, -0.003474361969435291, -0.003474361969435291, -0.003474361969435291, -0.003474361969435291, -0.003474361969435291, -0.003474361969435291, -0.010423196356281084, -0.010423196356281084, -0.010423196356281084, -0.010423196356281084, -0.010423196356281084, -0.010423196356281084, 0.03242763609333754, 0.03242763609333754, 0.03242763609333754, 0.03242763609333754, 0.03242763609333754, 0.03242763609333754, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.024320754681997014, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, 0.002316259720952729, -0.006948779162858076, -0.006948779162858076, -0.006948779162858076, -0.006948779162858076, -0.006948779162858076, -0.006948779162858076, 0.009265038883810806, 0.009265038883810806, 0.009265038883810806, 0.009265038883810806, 0.009265038883810806, 0.009265038883810806, -0.03242769131732537, -0.03242769131732537, -0.03242769131732537, -0.03242769131732537, -0.03242769131732537, -0.03242769131732537, 0.03474389581429005, 0.03474389581429005, 0.03474389581429005, 0.03474389581429005, 0.03474389581429005, 0.03474389581429005, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, 0.01621381804666866, -0.01737197551913905, -0.01737197551913905, -0.01737197551913905, -0.01737197551913905, -0.01737197551913905, -0.01737197551913905, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04748335189152375, 0.04748335189152375, 0.04748335189152375, 0.04748335189152375, 0.04748335189152375, 0.04748335189152375, 0.017372030743126765, 0.017372030743126765, 0.017372030743126765, 0.017372030743126765, 0.017372030743126765, 0.017372030743126765, 0.030111376372384813, 0.030111376372384813, 0.030111376372384813, 0.030111376372384813, 0.030111376372384813, 0.030111376372384813, 0.020846337488574118, 0.020846337488574118, 0.020846337488574118, 0.020846337488574118, 0.020846337488574118, 0.020846337488574118, 0, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, -0.01505571579818632, -0.01505571579818632, -0.01505571579818632, -0.01505571579818632, -0.01505571579818632, -0.01505571579818632, 0.02432064423402136, 0.02432064423402136, 0.02432064423402136, 0.02432064423402136, 0.02432064423402136, 0.02432064423402136, -0.00810699185931607, -0.00810699185931607, -0.00810699185931607, -0.00810699185931607, -0.00810699185931607, -0.00810699185931607, -0.022004439737056458, -0.022004439737056458, -0.022004439737056458, -0.022004439737056458, -0.022004439737056458, -0.022004439737056458, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, -0.009265038883810695, 0.010423085908305318, 0.010423085908305318, 0.010423085908305318, 0.010423085908305318, 0.010423085908305318, 0.010423085908305318, -0.005790566466399971, -0.005790566466399971, -0.005790566466399971, -0.005790566466399971, -0.005790566466399971, -0.005790566466399971, 0.050957824308934474, 0.050957824308934474, 0.050957824308934474, 0.050957824308934474, 0.050957824308934474, 0.050957824308934474, -0.0011581574724702781, -0.0011581574724702781, -0.0011581574724702781, -0.0011581574724702781, -0.0011581574724702781, -0.0011581574724702781, -0.02200443973705657, -0.02200443973705657, -0.02200443973705657, -0.02200443973705657, -0.02200443973705657, -0.02200443973705657, -0.027795116651432195, -0.027795116651432195, -0.027795116651432195, -0.027795116651432195, -0.027795116651432195, -0.027795116651432195, 0.01737192029515111, 0.01737192029515111, 0.01737192029515111, 0.01737192029515111, 0.01737192029515111, 0.01737192029515111, -0.002316204496964902, -0.002316204496964902, -0.002316204496964902, -0.002316204496964902, -0.002316204496964902, -0.002316204496964902, 0.06485527218667497, 0.06485527218667497, 0.06485527218667497, 0.06485527218667497, 0.06485527218667497, 0.06485527218667497, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, 0.019688235240091667, 0.019688235240091667, 0.019688235240091667, 0.019688235240091667, 0.019688235240091667, 0.019688235240091667, 0.023162597209526736, 0.023162597209526736, 0.023162597209526736, 0.023162597209526736, 0.023162597209526736, 0.023162597209526736, -0.011581243380775597, -0.011581243380775597, -0.011581243380775597, -0.011581243380775597, -0.011581243380775597, -0.011581243380775597, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.07991098798486129, -0.07991098798486129, -0.07991098798486129, -0.07991098798486129, -0.07991098798486129, -0.07991098798486129, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, 0.009265038883810695, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, -0.0069488343868459035, 0.01158135382875114, 0.01158135382875114, 0.01158135382875114, 0.01158135382875114, 0.01158135382875114, 0.01158135382875114, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, -0.06253906768971007, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.039376470480183334, 0.039376470480183334, 0.039376470480183334, 0.039376470480183334, 0.039376470480183334, 0.039376470480183334, 0.04516714739455896, 0.04516714739455896, 0.04516714739455896, 0.04516714739455896, 0.04516714739455896, 0.04516714739455896, 0.0023163149449405007, 0.0023163149449405007, 0.0023163149449405007, 0.0023163149449405007, 0.0023163149449405007, 0.0023163149449405007, -0.017371920295151166, -0.017371920295151166, -0.017371920295151166, -0.017371920295151166, -0.017371920295151166, -0.017371920295151166, 0.02200443973705657, 0.02200443973705657, 0.02200443973705657, 0.02200443973705657, 0.02200443973705657, 0.02200443973705657, 0.01158135382875125, 0.01158135382875125, 0.01158135382875125, 0.01158135382875125, 0.01158135382875125, 0.01158135382875125, 0.04632519441905353, 0.04632519441905353, 0.04632519441905353, 0.04632519441905353, 0.04632519441905353, 0.04632519441905353, 0.0011580470244945962, 0.0011580470244945962, 0.0011580470244945962, 0.0011580470244945962, 0.0011580470244945962, 0.0011580470244945962, 0.04864150936399403, 0.04864150936399403, 0.04864150936399403, 0.04864150936399403, 0.04864150936399403, 0.04864150936399403, -0.0069487239388702216, -0.0069487239388702216, -0.0069487239388702216, -0.0069487239388702216, -0.0069487239388702216, -0.0069487239388702216, 0.08454350742676667, 0.08454350742676667, 0.08454350742676667, 0.08454350742676667, 0.08454350742676667, 0.08454350742676667, -0.04285083244961846, -0.04285083244961846, -0.04285083244961846, -0.04285083244961846, -0.04285083244961846, -0.04285083244961846, -0.047483351891523806, -0.047483351891523806, -0.047483351891523806, -0.047483351891523806, -0.047483351891523806, -0.047483351891523806, -0.01853007776762139, -0.01853007776762139, -0.01853007776762139, -0.01853007776762139, -0.01853007776762139, -0.01853007776762139, 0.016213873270656598, 0.016213873270656598, 0.016213873270656598, 0.016213873270656598, 0.016213873270656598, 0.016213873270656598, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, 0.0034743619694351247, 0.0034743619694351247, 0.0034743619694351247, 0.0034743619694351247, 0.0034743619694351247, 0.0034743619694351247, 0.0023162044969648465, 0.0023162044969648465, 0.0023162044969648465, 0.0023162044969648465, 0.0023162044969648465, 0.0023162044969648465, 0.010423085908305346, 0.010423085908305346, 0.010423085908305346, 0.010423085908305346, 0.010423085908305346, 0.010423085908305346, 0.006948834386845848, 0.006948834386845848, 0.006948834386845848, 0.006948834386845848, 0.006948834386845848, 0.006948834386845848, -0.0428508324496184, -0.0428508324496184, -0.0428508324496184, -0.0428508324496184, -0.0428508324496184, -0.0428508324496184, 0.03126958906884286, 0.03126958906884286, 0.03126958906884286, 0.03126958906884286, 0.03126958906884286, 0.03126958906884286, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, -0.005790676914375625, 0.013897668773691696, 0.013897668773691696, 0.013897668773691696, 0.013897668773691696, 0.013897668773691696, 0.013897668773691696, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, 0.025478912154467293, -0.008106881411340472, -0.008106881411340472, -0.008106881411340472, -0.008106881411340472, -0.008106881411340472, -0.008106881411340472, -0.016213873270656542, -0.016213873270656542, -0.016213873270656542, -0.016213873270656542, -0.016213873270656542, -0.016213873270656542, -0.02547880170649164, -0.02547880170649164, -0.02547880170649164, -0.02547880170649164, -0.02547880170649164, -0.02547880170649164, -0.017372030743126807, -0.017372030743126807, -0.017372030743126807, -0.017372030743126807, -0.017372030743126807, -0.017372030743126807, -0.028953274123902376, -0.028953274123902376, -0.028953274123902376, -0.028953274123902376, -0.028953274123902376, -0.028953274123902376, -0.03474406148625361, -0.03474406148625361, -0.03474406148625361, -0.03474406148625361, -0.03474406148625361, -0.03474406148625361, 0.07527857899093157, 0.07527857899093157, 0.07527857899093157, 0.07527857899093157, 0.07527857899093157, 0.07527857899093157, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, -0.08222741337777745, -0.08222741337777745, -0.08222741337777745, -0.08222741337777745, -0.08222741337777745, -0.08222741337777745, -0.08106925590530717, -0.08106925590530717, -0.08106925590530717, -0.08106925590530717, -0.08106925590530717, -0.08106925590530717, 0.003474472417410744, 0.003474472417410744, 0.003474472417410744, 0.003474472417410744, 0.003474472417410744, 0.003474472417410744, -0.03126947862086724, -0.03126947862086724, -0.03126947862086724, -0.03126947862086724, -0.03126947862086724, -0.03126947862086724, 0.003474472417410779, 0.003474472417410779, 0.003474472417410779, 0.003474472417410779, 0.003474472417410779, 0.003474472417410779, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, 0.017371920295151166, -0.018530077767621417, -0.018530077767621417, -0.018530077767621417, -0.018530077767621417, -0.018530077767621417, -0.018530077767621417, 0.011581353828751195, 0.011581353828751195, 0.011581353828751195, 0.011581353828751195, 0.011581353828751195, 0.011581353828751195, -0.013897668773691696, -0.013897668773691696, -0.013897668773691696, -0.013897668773691696, -0.013897668773691696, -0.013897668773691696, 0.005790676914375598, 0.005790676914375598, 0.005790676914375598, 0.005790676914375598, 0.005790676914375598, 0.005790676914375598, 0.08917613731664763, 0.08917613731664763, 0.08917613731664763, 0.08917613731664763, 0.08917613731664763, 0.08917613731664763, -0.01505560535021068, -0.01505560535021068, -0.01505560535021068, -0.01505560535021068, -0.01505560535021068, -0.01505560535021068, -0.031269589068842876, -0.031269589068842876, -0.031269589068842876, -0.031269589068842876, -0.031269589068842876, -0.031269589068842876, -0.028953274123902382, -0.028953274123902382, -0.028953274123902382, -0.028953274123902382, -0.028953274123902382, -0.028953274123902382, -0.01621387327065657, -0.01621387327065657, -0.01621387327065657, -0.01621387327065657, -0.01621387327065657, -0.01621387327065657, 0.004632519441905375, 0.004632519441905375, 0.004632519441905375, 0.004632519441905375, 0.004632519441905375, 0.004632519441905375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.024320754681997042, -0.006948834386845848, -0.006948834386845848, -0.006948834386845848, -0.006948834386845848, -0.006948834386845848, -0.006948834386845848, 0, 0.002316314944940473, 0.002316314944940473, 0.002316314944940473, 0.002316314944940473, 0.002316314944940473, 0.002316314944940473, -0.02895327412390239, -0.02895327412390239, -0.02895327412390239, -0.02895327412390239, -0.02895327412390239, -0.02895327412390239, -0.03474395103827798, -0.03474395103827798, -0.03474395103827798, -0.03474395103827798, -0.03474395103827798, -0.03474395103827798, 0.005790787362351231, 0.005790787362351231, 0.005790787362351231, 0.005790787362351231, 0.005790787362351231, 0.005790787362351231, 0, -0.061380799769264235, -0.061380799769264235, -0.061380799769264235, -0.061380799769264235, -0.061380799769264235, -0.061380799769264235, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, 0.03358579356580771, -0.0023162044969648465, -0.0023162044969648465, -0.0023162044969648465, -0.0023162044969648465, -0.0023162044969648465, -0.0023162044969648465, 0.010423196356280959, 0.010423196356280959, 0.010423196356280959, 0.010423196356280959, 0.010423196356280959, 0.010423196356280959, 0.0416926749771482, 0.0416926749771482, 0.0416926749771482, 0.0416926749771482, 0.0416926749771482, 0.0416926749771482, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.008106881411340458, 0.005790566466399971, 0.005790566466399971, 0.005790566466399971, 0.005790566466399971, 0.005790566466399971, 0.005790566466399971, 0.09265038883810708, 0.09265038883810708, 0.09265038883810708, 0.09265038883810708, 0.09265038883810708, 0.09265038883810708, 0.09380854631057739, 0.09380854631057739, 0.09380854631057739, 0.09380854631057739, 0.09380854631057739, 0.09380854631057739, 0.02316270765750239, 0.02316270765750239, 0.02316270765750239, 0.02316270765750239, 0.02316270765750239, 0.02316270765750239, -0.025479022602442836, -0.025479022602442836, -0.025479022602442836, -0.025479022602442836, -0.025479022602442836, -0.025479022602442836, 0.20614739128472737, 0.20614739128472737, 0.20614739128472737, 0.20614739128472737, 0.20614739128472737, 0.20614739128472737, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.02663718007491314, 0.0521157608854535, 0.0521157608854535, 0.0521157608854535, 0.0521157608854535, 0.0521157608854535, 0.0521157608854535, 0.1366594892081715, 0.1366594892081715, 0.1366594892081715, 0.1366594892081715, 0.1366594892081715, 0.1366594892081715, -0.07875294096036667, -0.07875294096036667, -0.07875294096036667, -0.07875294096036667, -0.07875294096036667, -0.07875294096036667, 0.013897447877740499, 0.013897447877740499, 0.013897447877740499, 0.013897447877740499, 0.013897447877740499, 0.013897447877740499, 0.06717158713161553, 0.06717158713161553, 0.06717158713161553, 0.06717158713161553, 0.06717158713161553, 0.06717158713161553, 0.006948944834821447, 0.006948944834821447, 0.006948944834821447, 0.006948944834821447, 0.006948944834821447, 0.006948944834821447, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, -0.03011143159637264, 0.23046803551874873, 0.23046803551874873, 0.23046803551874873, 0.23046803551874873, 0.23046803551874873, 0.23046803551874873, -0.11349678155066911, -0.11349678155066911, -0.11349678155066911, -0.11349678155066911, -0.11349678155066911, -0.11349678155066911, 0.03011121070042133, 0.03011121070042133, 0.03011121070042133, 0.03011121070042133, 0.03011121070042133, 0.03011121070042133, -0.05559023330286417, -0.05559023330286417, -0.05559023330286417, -0.05559023330286417, -0.05559023330286417, -0.05559023330286417, 0.027795116651432306, 0.027795116651432306, 0.027795116651432306, 0.027795116651432306, 0.027795116651432306, 0.027795116651432306, -0.04169278542512389, -0.04169278542512389, -0.04169278542512389, -0.04169278542512389, -0.04169278542512389, -0.04169278542512389, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, -0.2362586019851487, 0.14360821314704153, 0.14360821314704153, 0.14360821314704153, 0.14360821314704153, 0.14360821314704153, 0.14360821314704153, 0.11812941144055, 0.11812941144055, 0.11812941144055, 0.11812941144055, 0.11812941144055, 0.11812941144055, 0.07412053196643709, 0.07412053196643709, 0.07412053196643709, 0.07412053196643709, 0.07412053196643709, 0.07412053196643709, 0.07412031107048578, 0.07412031107048578, 0.07412031107048578, 0.07412031107048578, 0.07412031107048578, 0.07412031107048578, 0.06022286319274506, 0.06022286319274506, 0.06022286319274506, 0.06022286319274506, 0.06022286319274506, 0.06022286319274506, 0.24783995581389995, 0.24783995581389995, 0.24783995581389995, 0.24783995581389995, 0.24783995581389995, 0.24783995581389995, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.07643662601542633, -0.1505569370859119, -0.1505569370859119, -0.1505569370859119, -0.1505569370859119, -0.1505569370859119, -0.1505569370859119, 0.09033429478911792, 0.09033429478911792, 0.09033429478911792, 0.09033429478911792, 0.09033429478911792, 0.09033429478911792, 0.09728301872798806, 0.09728301872798806, 0.09728301872798806, 0.09728301872798806, 0.09728301872798806, 0.09728301872798806, -0.011581132932799942, -0.011581132932799942, -0.011581132932799942, -0.011581132932799942, -0.011581132932799942, -0.011581132932799942, -0.12507813537942014, -0.12507813537942014, -0.12507813537942014, -0.12507813537942014, -0.12507813537942014, -0.12507813537942014, -0.034743840590302444, -0.034743840590302444, -0.034743840590302444, -0.034743840590302444, -0.034743840590302444, -0.034743840590302444, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.226993563101338, 0.226993563101338, 0.226993563101338, 0.226993563101338, 0.226993563101338, 0.226993563101338, -0.10191542772191786, -0.10191542772191786, -0.10191542772191786, -0.10191542772191786, -0.10191542772191786, -0.10191542772191786, 0.020846392712561945, 0.020846392712561945, 0.020846392712561945, 0.020846392712561945, 0.020846392712561945, 0.020846392712561945, 0.2223611541074082, 0.2223611541074082, 0.2223611541074082, 0.2223611541074082, 0.2223611541074082, 0.2223611541074082, 0.08106903500935592, 0.08106903500935592, 0.08106903500935592, 0.08106903500935592, 0.08106903500935592, 0.08106903500935592, -0.07412053196643686, -0.07412053196643686, -0.07412053196643686, -0.07412053196643686, -0.07412053196643686, -0.07412053196643686, -0.004632408993929582, -0.004632408993929582, -0.004632408993929582, -0.004632408993929582, -0.004632408993929582, -0.004632408993929582, 0, 0.29879755922688345, 0.29879755922688345, 0.29879755922688345, 0.29879755922688345, 0.29879755922688345, 0.29879755922688345, -0.09033407389316661, -0.09033407389316661, -0.09033407389316661, -0.09033407389316661, -0.09033407389316661, -0.09033407389316661, -0.002316314944940334, -0.002316314944940334, -0.002316314944940334, -0.002316314944940334, -0.002316314944940334, -0.002316314944940334, 0.5257915641201238, 0.5257915641201238, 0.5257915641201238, 0.5257915641201238, 0.5257915641201238, 0.5257915641201238, 0.15055737887781362, 0.15055737887781362]
    Final amount of money: 1002.9013651896595
:::

::: {.output .execute_result execution_count="25"}
    <function matplotlib.pyplot.legend(*args, **kwargs)>
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/7b704b4008c8c8555a6d8c5d3360a47961633bc6.png)
:::
:::

::: {.cell .code execution_count="26" colab="{\"height\":682,\"base_uri\":\"https://localhost:8080/\"}" id="Cm8N1tJvcnJQ" outputId="fc59eb70-ada9-4afe-8258-d7e52f624a68"}
``` {.python}
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def BBANDS(data, window):
    # Compute the Bollinger Bands 
    MA = data.Close.rolling(window=window).mean()
    SD = data.Close.rolling(window=window).std()
    data['MiddleBand'] = MA
    data['UpperBand'] = MA + (2 * SD) 
    data['LowerBand'] = MA - (2 * SD)
    return data

# Retrieve the stock data from Yahoo finance
data = yf.download("GOOG", start="2020-01-01", end="2020-12-31")

# Define the window size for the Bollinger Bands
n = 50

# Compute the Bollinger Bands for the stock using the 50-day Moving average
BBANDS = BBANDS(data, n)


buy_signal=[]
sell_signal=[]
# Look for buy and sell signals using the Bollinger Bands
for i in range(len(data)):
    # If the stock price touches or moves outside the upper band, this can be a sell signal
    if data['Close'][i] >= BBANDS['UpperBand'][i]:
        sell_signal.append(i)
        print("Sell signal at index", i)
    # If the stock price touches or moves outside the lower band, this can be a buy signal
    elif data['Close'][i] <= BBANDS['LowerBand'][i]:
        print("Buy signal at index", i)
        buy_signal.append(i)

# Plot the stock data and Bollinger Bands
plt.plot(data['Close'], label='Close')
plt.plot(BBANDS['MiddleBand'], label='Middle Band')
plt.plot(BBANDS['UpperBand'], label='Upper Band')
plt.plot(BBANDS['LowerBand'], label='Lower Band')
plt.plot(data['Close'], '^', markersize=10, color='m', label = 'buying signal', markevery = buy_signal)
plt.plot(data['Close'], 'v', markersize=10, color='k', label = 'selling signal', markevery = sell_signal)
plt.legend()
plt.show()
```

::: {.output .stream .stdout}
    [*********************100%***********************]  1 of 1 completed
    Buy signal at index 49
    Buy signal at index 50
    Buy signal at index 51
    Buy signal at index 52
    Buy signal at index 53
    Buy signal at index 54
    Buy signal at index 55
    Sell signal at index 130
    Sell signal at index 131
    Sell signal at index 137
    Sell signal at index 138
    Sell signal at index 139
    Sell signal at index 163
    Sell signal at index 164
    Sell signal at index 165
    Sell signal at index 166
    Sell signal at index 168
    Sell signal at index 169
    Sell signal at index 213
    Sell signal at index 214
    Sell signal at index 215
    Sell signal at index 216
    Sell signal at index 220
:::

::: {.output .display_data}
![](vertopal_b218f94d8f6a47f4ab3906e9d046bcf5/e6c6b7303e0fc8a8480912618daf40ea26a39ec7.png)
:::
:::

::: {.cell .markdown id="gyO3-l73rX_s"}
\#Result and Discussion
:::

::: {.cell .markdown id="vCIZwV2Vraw8"}
Data structures are consequently introduced and displayed on the graph.
These time series data structures were used for future ARIMA
predictions, and the resulting R2 score was noted. As a result of the
failure of the forecasts made in this section, new models were
emphasized. These new models use the reinforcement learning technique to
purchase and sell on the provided graph. Trades were executed in
accordance with the defined inductor structures that will provide
trading signals to these models. The following are the results for the
trades executed by the agents linked to these models:

Akfen= %119,2087

Sekerbank = %109,3764

Ereğli = %90,087

Şişe Cam = %51,4276

Koza Altın = %130,61

Vestel = %114,17

an increase was observed. However, if we consider the min and max values
of these shares in certain periods;

Akfen = from 0,7576 to 8,86

Şekerbank = from 0,65 to 5,25

Ereğli = from 6,14 to 42,66

Şişe Cam = from 3,71 to 40,36

Koza Altın = from 5,37 to 57,05

Vestel = from 5,26 to 66,6 was observed.

The study was successful based on these findings, but it\'s possible
that this is a function of the fact that it was conducted at a time when
the stock market was rising. Different stocks must be trained in models
that are bearish, stable, and bullish for the long, short, and medium
terms in order to accurately calculate the success rate.
:::

::: {.cell .markdown id="UC4qT4y3rbqo"}
\#Future Works
:::

::: {.cell .markdown id="ePhAXKcsre-Y"}
The creation of a new agent is the most significant task that can be
accomplished for this project in the future. In order to make an
appropriate transaction, this agent will examine both the depth data and
the end-of-day closing data. The freshly generated agent\'s evolutionary
approach may be its learning strategy. You could also choose to believe
rumors. In an NLP project, a distinct feature with the articles written
on the shares on particular websites can be added to produce an
acceptable algorithm (tradingview, twitter ..).
:::

::: {.cell .markdown id="rCwhX7ocrfqv"}
\#References
:::

::: {.cell .markdown id="M-Cvf5XBrhyG"}
-   Nesrin Koç Ustalı, Nedret Tosun, Ömür Tosun.(2020).Makine Öğrenmesi
    Teknikleri ile Hisse Senedi Fiyat Tahmini
-   Mehmet ÖZÇALICI, Yücel AYRIÇAY.(2016).BİLGİ İŞLEMSEL ZEKA YÖNTEMLERİ
    İLE HİSSE SENEDİ FİYAT TAHMİNİ: BİST UYGULAMASI
-   Tamerlan Mashadihasanli.(2022).Stock Market Price Forecasting Using
    the Arima Model: an Application to Istanbul, Turkiye
-   Stock-Prediction-Models.(2021) .github.com
    :github.com/huseinzol05/Stock-Prediction-Models
-   Rohan Kumar.(2021).Python for stock analysis.
    medium.com/analytics-vidhya/python-for-stock-analysis-fcff252ca559
-   Shadap Hussein.Modelling and Evaluation - Time Series Forecasting
    using PyCaret.
    developers.refinitiv.com/en/article-catalog/article/modelling-and-evaluation-using-pycaret-on-time-series-data
-   Chainika Thakar, Danish Khajuria.(2022). Building Technical
    Indicators in Python.
    blog.quantinsti.com/build-technical-indicators-in-python/
-   Rohan Kumar.(2021). How to Build Stock Technical Indicators with
    Python.
    medium.com/analytics-vidhya/how-to-build-stock-technical-indicators-with-python-7a0c5b665285
:::
