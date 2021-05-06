# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk(Dataplore):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import datetime as dt
from itertools import chain

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

vacc1=pd.read_csv("http://api.covid19india.org/csv/latest/cowin_vaccine_data_statewise.csv")
vacc1.fillna(0)
vacc1["Updated On"]=pd.to_datetime(vacc1["Updated On"],infer_datetime_format=True)
vacc1.drop(["AEFI"],inplace=True,axis=1)
#display(vacc1.columns)
#display(vacc1)

vacc=pd.read_csv("http://api.covid19india.org/csv/latest/vaccine_doses_statewise.csv")
vacc.fillna(0)
vacc.index=vacc["State"]
vacc=vacc.T
vacc.drop(["State"],inplace=True)
vacc.index=pd.to_datetime(vacc.index,infer_datetime_format=True)
#display(vacc)
#display(vacc.columns)

def f(x):
    display(x)
    return x

C = vacc1["State"].unique()
P = interactive(f, x=widgets.Dropdown(options=C,value='India',description='State:',disabled=False))
print("Select a state to view vaccination data:")
display(P)

#Total doses administered nationwide
res=P.result
State=vacc1[vacc1["State"]==res]
D=State["Updated On"]
X=State["Total Doses Administered"]
plt.figure(figsize=(16,8))
plt.bar(D,X)
plt.title("{}: Total doses administered".format(res))
plt.grid()
plt.show()

#First done vs fully vaccinated
X=State["First Dose Administered"]
Y=State["Second Dose Administered"]
P=Y/X*100
plt.figure(figsize=(16,8))
plt.plot(D,X,label="First dose")
plt.plot(D,Y,label="Second dose")
plt.title("First dose done vs Second dose done")
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
plt.plot(P,color="purple")
plt.title("Percentage of people fully vaccinated")
plt.grid()
plt.show()

#Male vs Female vaccinations
A=State[State.index==(State.index).max()-3]["Male(Individuals Vaccinated)"].values
B=State[State.index==(State.index).max()-3]["Female(Individuals Vaccinated)"].values
new=list(chain.from_iterable([A,B]))
plt.pie(new,labels=["Male","Female"],radius=2,shadow=True,autopct='%1.1f%%',explode = [0,0.1])
plt.title("Vaccinations completed")
plt.legend()
plt.show()

#Covaxin vs Covishield
A=State["Total Covaxin Administered"].diff()
B=State["Total CoviShield Administered"].diff()
X=State[State.index==(State.index).max()-3]["Total Covaxin Administered"].values
Y=State[State.index==(State.index).max()-3]["Total CoviShield Administered"].values

plt.figure(figsize=(16,8))
plt.bar(D,B,label="CoviShield",color="orange")
plt.bar(D,A,label='Covaxin',color="blue")
plt.title("Daily CoviShield vs Covaxin")
plt.legend(loc='upper right')
plt.grid()
plt.show()

new=list(chain.from_iterable([X,Y]))
plt.pie(new,labels=["Covaxin","CoviShield"],radius=2,shadow=True,autopct='%1.1f%%',explode = [0,0.2])
plt.title("Vaccine brand")
plt.legend()
plt.show()

def f(x):
    display(x)
    return x

C = vacc.columns[1:38]
P = interactive(f, x=widgets.Dropdown(options=C,value='Bihar',description='State:',disabled=False))
print("Select a state to view vaccination data:")
display(P)

State = P.result
X=vacc["Total"] #Total Vaccinations
Y=vacc[State]  #State total vaccinations
DX=X.diff() #Daily vaccinations
DY=Y.diff()  #State Vaccinations
D=vacc.index

plt.figure(figsize=(16,8))
plt.bar(D,abs(DY))
plt.title("Daily {} Vaccinations".format(State))
plt.grid()
plt.show()

state_names=["India",'Andaman Nicobar', 'Andhra P', 'Arunachal P',
       'Assam', 'Bihar', 'Chandigarh', 'Chhattisgarh',
       'Daman and Diu', 'Delhi', 'Goa', 'Gujarat',
       'Haryana', 'Himachal P', 'Jammu Kashmir', 'Jharkhand',
       'Karnataka', 'Kerala', 'Ladakh', 'Lakshadweep', 'Madhya P',
       'Maharashtra', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Odisha',
       'Puducherry', 'Punjab', 'Rajasthan', 'Sikkim', 'Tamil Nadu',
       'Telangana', 'Tripura', 'Uttar P', 'Uttarakhand', 'West Bengal']

st_df=pd.DataFrame()
for st in vacc.columns:
    X=vacc[st][len(vacc)-2]
    #print("{}: {}".format(st,X))
    st_df=st_df.append([X])
    
st_df.index=vacc.columns
st_df.rename(index={"Total":"India"},inplace=True)
st_df.drop(["Miscellaneous"],inplace=True)
st_df.columns=["Vaccinations"]
sorted_st_df=st_df.sort_values(by=["Vaccinations"],ascending=False)

pop=pd.read_csv('State-census.csv')
pop["Population"]=pd.to_numeric(pop["Population"],errors="coerce")
pop["Density"]=pd.to_numeric(pop["Density"],errors="coerce")
pop["Growth Rate"]=pd.to_numeric(pop["Growth Rate"],errors="coerce")
display(pop.head())

#Adding columns to new dataset for vaccinations and population.
st_df["Population"]=pop["Population"].values
st_df["Urban Pop"]=pop["Urban Population"].values
st_df["Rural Pop"]=pop["Rural Population"].values
st_df["Sex Ratio"]=pop["Sex Ratio"].values
display(st_df.head())

plt.figure(figsize=(16,10))
sns.barplot(sorted_st_df["Vaccinations"][1:],sorted_st_df.index[1:],saturation=1,orient="h")
plt.title("Total Vaccinations",size=20)
plt.grid()
plt.xticks(rotation=90,size=12)
plt.tight_layout()
plt.show()

# Top 5 vaccinated states
plt.figure(figsize=(10,6))
plt.plot(sorted_st_df[1:].head())
plt.title("Maximun vaccinated States")
plt.grid()

gender_df=pd.DataFrame()
brand_df=pd.DataFrame()
for st in vacc1["State"].unique():
    sp=vacc1[vacc1["State"]==st]
    X=sp[sp.index==(sp.index).max()-3]["Male(Individuals Vaccinated)"].values
    Y=sp[sp.index==(sp.index).max()-3]["Female(Individuals Vaccinated)"].values
    Covishield=sp[sp.index==(sp.index).max()-3]["Total CoviShield Administered"].values
    Covaxin=sp[sp.index==(sp.index).max()-3]["Total Covaxin Administered"].values
    gender_df=gender_df.append(list(zip(X,Y)))
    brand_df=brand_df.append(list(zip(Covaxin,Covishield)))
    
gender_df.columns=["Male","Female"]
brand_df.columns=["Covaxin","Covishield"]
gender_df.index=state_names
brand_df.index=state_names
#gender_df.sort_values(by=["Male"],ascending=False,inplace=True)

#Comparison of gender and brands
Gen = go.Figure(data=[go.Bar(x=gender_df.index[1:],y=gender_df["Male"][1:],    #Trace 1
                                   name="Male",
                                   marker = dict(color = 'rgba(0, 0, 255,1)'),
                           ),
                     go.Bar(x=gender_df.index[1:],y=gender_df["Female"][1:],name="Female",
                            marker=dict(color = 'rgba(255, 0, 0,1)')),
                    ]   
              )
Gen.update_xaxes(
    tickangle = 90,
    title_font = {"size": 1},
)
Gen.update_layout(
    height=600,
    margin=dict(l=20,r=20,b=50,t=50),
    title="Male vs Female",
    xaxis_title="State",
    yaxis_title="Vaccinations",
    legend_title="Gender: ",
    paper_bgcolor='#9ef7ad',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([dict(count=3,step="month",stepmode="backward"),
                          dict(count=6,step="month",stepmode="backward"),
                          dict(count=1,step="year",stepmode="backward")])
        ),
        rangeslider=dict(visible=True),
    )
)

Brand = go.Figure(data=[go.Bar(x=brand_df.index[1:],y=brand_df["Covaxin"][1:],    #Trace 1
                                   name="Covaxin",
                                   marker = dict(color = 'rgba(0, 0, 255,1)'),
                           ),
                     go.Bar(x=brand_df.index[1:],y=brand_df["Covishield"][1:],
                            name="Covishield",
                            marker=dict(color = 'rgba(255, 0, 0,1)')),
                    ]   
              )
Brand.update_xaxes(
    tickangle = 90,
    title_font = {"size": 1},
)
Brand.update_layout(
    height=600,
    margin=dict(l=20,r=20,b=50,t=50),
    title="Covaxin vs Covishield",
    xaxis_title="State",
    yaxis_title="Vaccinations",
    legend_title="Brand: ",
    paper_bgcolor='#9ef7ad',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([dict(count=3,step="month",stepmode="backward"),
                          dict(count=6,step="month",stepmode="backward"),
                          dict(count=1,step="year",stepmode="backward")])
        ),
        rangeslider=dict(visible=True),
    )
)

Gen.show()
Brand.show()

#Brand Wise:

#All the states have used CoviShield as their primary vaccine, clearly. Overall, Covishield takes up almost 90% of the total vaccinations in India while Covaxin take up a small value of 10%

#Sex Ratio:

#The sex ratio is the ratio of males to females in the population (normalized to 1000). Shown below is the sex ratio for each state in India.

#It basically shows the number of females per 1000 males.

plt.figure(figsize=(16,6))
plt.plot(pop["State"],pop["Sex Ratio"])
plt.scatter(pop["State"],pop["Sex Ratio"])
plt.ylabel("Number of females")
plt.xticks(rotation=90)
plt.xticks(rotation=90)
plt.title("Sex Ratio State-wise")
plt.grid()
plt.show()

#As we can see, most of the states have a higher number of males (i.e. a sex ratio <1000). Only kerala and puduchery have more females.

#Population
plt.figure(figsize=(18,6))
sns.barplot(pop["State"][:36],pop["Population"][1:]*100)
plt.xticks(rotation=90)
plt.grid()
plt.show()

#Population Density
plt.figure(figsize=(16,8))
plt.plot(pop["State"][:36],pop["Density"][:36],color="purple")
plt.title("Population Density")
plt.xticks(rotation=90)
plt.grid()
plt.show()

ur_per=pop["Urban Population"][:36]*100/pop["Population"][:36]
rur_per=pop["Rural Population"][:36]*100/pop["Population"][:36]

plt.figure(figsize=(16,8))
plt.plot(pop["State"][:36],rur_per,label="Rural")
plt.plot(pop["State"][:36],ur_per,label="Urban")
plt.scatter(pop["State"][:36],rur_per)
plt.scatter(pop["State"][:36],ur_per)
plt.title("Urban vs Rural population (%)")
plt.xticks(rotation=90)
plt.legend()
plt.grid()
plt.show()

# **VACCINATION STATUS**


plt.figure(figsize=(18,6))
plt.bar(st_df.index,(st_df["Vaccinations"]*100/st_df["Population"]),color="#D2B48C")
plt.title("Vaccinations / Population")
plt.xticks(rotation=90) 
plt.grid()
plt.show()

# **CORELATION B/W RURAL && URBAN**


#Vaccinations in urban and rural areas
f, ax = plt.subplots(1, 2, figsize = (20, 6))

sns.regplot(st_df["Urban Pop"][:36],st_df["Vaccinations"][:36],ax=ax[0])
ax[0].grid()
ax[0].set_xlabel("Urban Population",fontSize=15)

sns.regplot(st_df["Rural Pop"][:36],st_df["Vaccinations"][:36],ax=ax[1])
ax[1].grid()
ax[1].set_xlabel("Rural Population",fontSize=15)

plt.show()

#Correlation between urban population and vaccination
newdf =st_df[['Vaccinations','Urban Pop','Rural Pop']]
correlation = newdf.corr()
sns.heatmap(correlation, cmap="Blues",linewidths=2,annot=True) 
plt.tight_layout()

# If the value is near ± 1, then it said to be a perfect correlation: as one variable increases, the other variable tends to also increase (if positive) or decrease (if negative). High degree: If the coefficient value lies between ± 0.50 and ± 1, then it is said to be a strong correlation. Hence a strong corelation can be seen in Rural Vs Urban Population




