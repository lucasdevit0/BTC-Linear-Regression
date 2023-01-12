# Author: Lucas Casa de Vito
# Date: 11.02.22
# BTC Price Analysis using Logarithmic Regression

from turtle import left, right
import pandas as pd
import numpy as np
import os
import scipy
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#Clear terminal for each run
os.system("cls")

# Import data
df = pd.read_csv("btc_marketprice.csv")

# Data processing
df = df.iloc[::-1]                          #reverse data order
df = df[ df['Value'] > 0]                   #exclude values < 0
df['Date'] = pd.to_datetime(df['Date'])
print(df)

# Regression fitting
def function(x,c1,c2):
    return c1*np.log(x) + c2

xdata = np.array([x+1 for x in range(len(df))]) #days
ydata = np.log(df['Value'])                     #log of values

popt, pcov = curve_fit(function,xdata,ydata,p0 = [3.8,-10])
print(popt)

fittedydata = function(xdata,popt[0],popt[1])


plt.title('BTC Price Chart (Log Scale)',fontsize=34)
plt.xlabel('Years',fontsize=34)
plt.ylabel('BTC Price (USD)',fontsize=34)
plt.grid(color='b', linestyle='--', linewidth=0.2)
#plt.minorticks_on()
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)


plt.semilogy(df['Date'],df['Value'],'k-')

colors = ['red','salmon','darkorange','orange','gold','greenyellow','green','olivedrab','limegreen']
count = 0
lin = np.linspace(-2,4,9)
for i in lin:
    plt.fill_between(df['Date'], np.exp(fittedydata + i-1 ), np.exp(fittedydata + 1), alpha = 0.2,color=colors[count])
    count = count + 1

count = 0
labels = ["Max bubble territory","SELL!","FOMO season","Is this a bubble?","HODL!","Still cheap","Accumulate","BUY!","Fire Sale!"]
for i in lin:
    plt.plot(df['Date'], np.exp(fittedydata), color=colors[len(colors)-(1+count)],alpha = 1 ,label=labels[count],linewidth = 1.2)
    count = count + 1

plt.plot(df['Date'], np.exp(fittedydata), color='black',alpha=0.4)


plt.ylim(bottom = 1)
plt.ylim(top = 500000)
plt.legend(loc=2,prop={'size': 35})

plt.show()
