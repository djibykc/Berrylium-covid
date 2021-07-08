#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""

LICENSE MIT

"""


# In[2]:


def nbWithSpaces(nb):
    str_nb = str(int(round(nb)))
    if(nb>100000):
        return str_nb[:3] + " " + str_nb[3:]
    elif(nb>10000):
        return str_nb[:2] + " " + str_nb[2:]
    elif(nb>1000):
        return str_nb[:1] + " " + str_nb[1:]
    else:
        return str_nb


# In[3]:


import pandas as pd
PATH = "../../"
import src.france.berryllium_france_data_management as data
import plotly.graph_objects as go
import locale
from datetime import datetime
locale.setlocale(locale.LC_ALL, 'fr_FR.UTF-8')
now = datetime.now()


# In[4]:


data.download_data()
df_tests =data.import_data_tests_sexe()
df_tests = df_tests[df_tests.cl_age90 == 0]
df_tests["P_rolling"] = df_tests["P"].rolling(window=7).mean()
df_tests


# In[5]:


data.download_data_variants()
df_variants = data.import_data_variants()
df_variants


# In[6]:


df_variants["jour"] = df_variants.semaine.apply(lambda x: x[11:]) 
df_variants = df_variants[df_variants.cl_age90==0]
df_variants


# In[7]:


fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=df_variants.Prc_susp_ABS,
        name="% souche classique (" + str(df_variants.Prc_susp_ABS.values[-1]).replace(".", ",") + " %)",
        showlegend=True,
    )
)

fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=df_variants.Prc_susp_501Y_V1,
        name="% variant UK (" + str(df_variants.Prc_susp_501Y_V1.values[-1]).replace(".", ",") + " %)",
    )
)

fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=df_variants.Prc_susp_IND,
        name="% variants indéterminés (" + str(df_variants.Prc_susp_IND.values[-1]).replace(".", ",") + " %)",
        showlegend=True,
    )
)

fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=df_variants.Prc_susp_501Y_V2_3,
        name="% variants SA + BZ (" + str(df_variants.Prc_susp_501Y_V2_3.values[-1]).replace(".", ",") + " %)",
        showlegend=True,
    )
)

fig.update_yaxes(ticksuffix="%")

fig.update_layout(
     title={
        'text': "Proportion de variants dans les tests positifs (en %)",
        'y':0.99,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
         'font': {'size': 30}
    },
    annotations = [
                    dict(
                        x=0.5,
                        y=1.1,
                        xref='paper',
                        yref='paper',
                        text='Mis à jour le {}. Données : Santé publique France. Auteur : @Djiby CASSE & Alpha SOW -'.format(now.strftime('%d %B')),
                        showarrow = False
                    )]
)
fig.write_image(PATH+"images/charts/france/{}.jpeg".format("variants_pourcent"), scale=2, width=1000, height=600)


# In[8]:


fig = go.Figure()
n_days = len(df_variants)

y=df_tests["P_rolling"].values[-n_days:] * df_variants.Prc_susp_501Y_V2_3.values/100
fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=y,
        name="<b>Variants SA + BZ </b><br>" + str(nbWithSpaces(y[-1])) + " (" + str(df_variants.Prc_susp_501Y_V2_3.values[-1]).replace(".", ",") + " %) ",
        showlegend=True,
        stackgroup='one'
    )
)

y=df_tests["P_rolling"].values[-n_days:] * df_variants.Prc_susp_IND.values/100
fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=y,
        name="<b>Variants indéterminés </b><br>" + str(nbWithSpaces(y[-1])) + " (" + str(df_variants.Prc_susp_IND.values[-1]).replace(".", ",") + " %) ",
        showlegend=True,
        stackgroup='one'
    )
)

y=df_tests["P_rolling"].values[-n_days:] * df_variants.Prc_susp_501Y_V1.values/100
fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=y,
        name="<b>Variant UK </b><br>" + str(nbWithSpaces(y[-1])).replace(".", ",") + " (" + str(df_variants.Prc_susp_501Y_V1.values[-1]).replace(".", ",") + " %) ",
        stackgroup='one'
    )
)

y=df_tests["P_rolling"].values[-n_days:] * df_variants.Prc_susp_ABS.values/100
fig.add_trace(
    go.Scatter(
        x=df_variants.jour,
        y=y,
        name="<b>Souche classique </b><br>" + str(nbWithSpaces(y[-1])) + " (" + str(df_variants.Prc_susp_ABS.values[-1]).replace(".", ",") + " %) ",
        showlegend=True,
        stackgroup='one'
    )
)


fig.update_yaxes(ticksuffix="")

fig.update_layout(
     title={
        'text': "Nombre de variants dans les cas détectés",
        'y':0.99,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
         'font': {'size': 30}
    },
    annotations = [
                    dict(
                        x=0.5,
                        y=1.1,
                        xref='paper',
                        yref='paper',
                        text='Mis à jour : {}. Données : Santé publique France. Auteur : @Djiby CASSE & Alpha SOW '.format(now.strftime('%d %B')),
                        showarrow = False
                    )]
)
fig.write_image(PATH+"images/charts/france/{}.jpeg".format("variants_nombre"), scale=2, width=1000, height=600)

