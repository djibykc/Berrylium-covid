#!/usr/bin/env python
# coding: utf-8

from multiprocessing import Pool
import requests
import pandas as pd
import math
import plotly.graph_objects as go
import plotly.express as px
import plotly
from plotly.subplots import make_subplots
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import imageio
import json
import locale
import src.france.berryllium_france_data_management as data
import numpy as np
import cv2

locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
colors = px.colors.qualitative.D3 + plotly.colors.DEFAULT_PLOTLY_COLORS + px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet
show_charts = False
PATH_STATS = "../../data/france/stats/"
PATH = "../../"
now = datetime.now()
# In[1]:


import pandas as pd


# In[2]:


confirmed_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
 
# Dataset is now stored in a Pandas Dataframe

recovered_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"

# Dataset is now stored in a Pandas Dataframe

death_cases_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
 
# Dataset is now stored in a Pandas Dataframe


# In[3]:


def get_n_melt_data(data_url,case_type):
    df = pd.read_csv(data_url)
    melted_df = df.melt(id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'])
    melted_df.rename(columns={"variable":"Date","value":case_type},inplace=True)
    return melted_df

def merge_data(confirm_df,recovered_df,deaths_df):
	new_df = confirm_df.join(recovered_df['Recovered']).join(deaths_df['Deaths'])
	return new_df


# In[4]:


confirm_df = get_n_melt_data(confirmed_cases_url,"Confirmed")
recovered_df = get_n_melt_data(recovered_cases_url,"Recovered")
deaths_df = get_n_melt_data(death_cases_url,"Deaths")


# In[5]:


confirm_df.tail()


# In[6]:


df = merge_data(confirm_df,recovered_df,deaths_df)


# In[7]:


df.head()


# In[8]:


df = df[['Country/Region','Date','Confirmed','Recovered','Deaths']].loc[df['Country/Region'] == 'France']


# In[9]:


df.tail()


# In[10]:


df = df.groupby("Date")[['Confirmed','Recovered', 'Deaths']].sum()


# In[11]:


df.tail()


# In[12]:


df_per_day = df.groupby("Date")[['Confirmed','Recovered', 'Deaths']].sum()
df_per_day.tail()


# In[13]:


df_per_day.plot(kind='line',figsize=(20,5))


# In[14]:


#Facebook Forecasting Library
import fbprophet


# In[15]:


# Model Initialize
from fbprophet import Prophet
m = Prophet()


# In[16]:


m.add_seasonality(name="monthly",period=30.5,fourier_order=5)


# In[17]:


# Split Dataset
df


# In[18]:


France_cases = df.reset_index()


# In[19]:


France_cases.head()


# In[20]:


France_cases.tail()


# In[21]:


confirmed_cases = France_cases[["Date","Confirmed"]]
recovered_cases = France_cases[["Date","Recovered"]]


# In[22]:


confirmed_cases.shape


# In[23]:


confirmed_cases.rename(columns={"Date":"ds","Confirmed":"y"},inplace=True)


# In[24]:


train = confirmed_cases[:12]
test = confirmed_cases[12:]


# In[25]:


train 


# In[26]:


test


# In[27]:


# Fit Model
m.fit(train)


# In[28]:


# Future Date
future_dates = m.make_future_dataframe(periods=200)


# In[29]:


future_dates


# In[30]:


# Prediction
prediction =  m.predict(future_dates)


# In[31]:


# Plot Prediction
m.plot(prediction)


# In[32]:


# Find Point/Dates For Change
from fbprophet.plot import add_changepoints_to_plot


# In[33]:


fig = m.plot(prediction)
c = add_changepoints_to_plot(fig.gca(),m,prediction)


# In[ ]:




