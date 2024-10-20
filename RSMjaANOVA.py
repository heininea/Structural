# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 12:12:50 2021

@author: localadmin
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as stats
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
# Jos dataan lisää sarakkeen, jossa on pelkkiä ykkösiä, saa b0 arvon suoraan mukaan
# Tosin helpompaa on luoda b vektori niin, että käyttää regr.coef_ b1...bn määrittämiseen
# ja regr.intercept_ kutsua b0 määrittämiseen ja lisää b0 b vektoriin komennolla
# numpy.insert(b, 0, regr.intercept_), missä 0 on vektorin b haluttu indeksi.

#pio.renderers.default = 'png'
pio.renderers.default = 'browser'


def LinearRegr(X,y):

    regr = linear_model.LinearRegression()
    regr.fit(X,y)
    b = regr.coef_[0]
    #b[0] = regr.intercept_
    b = np.insert(b, 0, regr.intercept_)
    #leikkaus = regr.intercept_
    #print(b)
    score = regr.score(X,y)
    #Lasketaan itse
    # pitää lisätä rivi ykkösiä X:ään
    X2 = np.insert(X, 0, np.ones(len(X)), axis=1)   
    ex = np.matmul(np.linalg.inv(np.matmul(np.transpose(X2),X2)),np.transpose(X2))
    b2 = np.matmul(ex,y)
    C = np.linalg.inv(np.matmul(np.transpose(X2),X2))
    yhat = np.matmul(X2,b2)
    r = y - yhat
    SSE = np.matmul(np.transpose(r),r)
    sSquared = SSE/(len(X) - (len(X[0])-1) - 1)
    
    return b, b2, SSE, yhat, sSquared, C, score

def multivariateRegr(X,y):
    
    X2 = X
    ex = np.matmul(np.linalg.inv(np.matmul(np.transpose(X2),X2)),np.transpose(X2))
    b2 = np.matmul(ex,y)
    C = np.linalg.inv(np.matmul(np.transpose(X2),X2))
    yhat = np.matmul(X2,b2)
    r = y - yhat
    SSE = np.matmul(np.transpose(r),r)
    sSquared = SSE/(len(X) - len(X[0]) - 1)
      
    return yhat, SSE, C, b2, sSquared, r

def coefOfDeterm(yhat, SSE, dime):
    # Calculates the coef of determination (R^2)
    
    M = np.identity(dime) - 1/dime*np.ones([dime,dime])
    
    SSR  = np.matmul(np.matmul(np.transpose(yhat), M), yhat)
    SST = SSR + SSE
       
    return SSR/SST

def anova(C, sSquared, dime, b):
     # Anova lol :D
     
    F = np.zeros(dime) 
    for i in range(dime):
        
        a = np.transpose(np.zeros(len(b)))
        a[i] = 1
        denom = np.matmul(np.transpose(a), np.matmul(C, a))
        nomi = (np.matmul(np.transpose(a), b))**2
        F[i] = 1/sSquared*nomi/denom
    
    p = 1 - stats.f.cdf(F, 1, dime)    
    return F, p
    
 #asdasdasd #Regression without intercept? 
 # mitä tulos oikeastaan tarkoittaa? osa kertoimista on "turhia" ja 
 # lopputulos voidaan ennustaa ilman niitä? 

def RMSCalc(b2,X,Y,Z):
    # Toimii
    RMS = b2[0] + b2[1]*X + b2[2]*Z + b2[3]*X*Y + b2[4]*X*Z + b2[5]*X*X + \
        b2[6]*Y*Y + b2[7]*Z*Z + b2[8]*X*X*Y + b2[9]*X*X*Z + b2[10]*Z*Z*X + b2[11]*X*X*Y*Y + b2[12]*X*X*Z*Z

    return RMS 

def plottaus(b2, opt):
    
    fMin = 1.5
    fMax = 2.25
    oMin = 6
    oMax = 11.2
    rMin = 70
    rMax = 130
    nPoints = 50
    opa = 0.5 # Opacity value 
    surfCount = 300 # Number of isosurfaces
  
    x = np.linspace(fMin, fMax, nPoints)
    y = np.linspace(oMin, oMax, nPoints)
    z = np.linspace(rMin, rMax, nPoints)
    
    X,Y,Z = np.meshgrid(x,y,z)
    
    
    #RMS = b2[0] + b2[1]*X + b2[2]*Z + b2[3]*X*Y + b2[4]*X*Z + b2[5]*X*X + \
       # b2[6]*Y*Y + b2[7]*Z*Z + b2[8]*X*X*Y + b2[9]*X*X*Z + b2[10]*Z*Z*X + b2[11]*X*X*Y*Y + b2[12]*X*X*Z*Z
    
    RMS = RMSCalc(b2, X, Y, Z)
    rmsFlat = RMS.flatten()
    
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=rmsFlat,
        isomin=np.min(rmsFlat),
        isomax=np.max(rmsFlat),
        opacity=opa, # needs to be small to see through all surfaces
        surface_count=surfCount, # needs to be a large number for good volume rendering
        colorscale = 'jet',
        colorbar=dict(
            lenmode='fraction',
            len=0.5, 
            thickness=50,
            tickfont=dict(family="Arial", size=24),
            dtick=25,
            tick0=40,
            nticks=4,
            ),

    ))

    vsbl = False

    fig.update_layout(
    autosize=False,
    template='simple_white',
    font_family="Arial",
    font_size=18,
    scene = dict( 
    bgcolor='white',
    xaxis = dict(
        showline=False,
        visible=vsbl,
        zeroline=False,
        showgrid=False,
        tickmode = 'linear',
        ticks="",
        tickwidth=1,
        tick0=1.5,
        dtick=0.25,
        title='Feed rate',
        titlefont=dict(family='Arial', size=24),
        ),
    yaxis = dict(
        showline=False,
        visible=vsbl,
        zeroline=False,
        showgrid=False,
        tickmode = 'linear',
        ticks="",
        tickwidth=1,
        tick0=6,
        dtick=1.7,
        title='Dressing overlap ratio',
        titlefont=dict(family='Arial', size=24),
        ),
    zaxis = dict(
        showline=False,
        visible=vsbl,
        zeroline=False,
        showgrid=False,
        tickmode = 'linear',
        ticks="",
        tickwidth=1,
        tick0=70,
        dtick=20,
        title='Workpiece rotational speed',
        titlefont=dict(family='Arial', size=24),
        )    
    ),
    plot_bgcolor='white',
    width=1200,
    height=1200,
    margin=dict(r=0, l=0, b=0, t=0, pad=100, autoexpand=True),
    
    )
    #pio.write_image(fig, "C:\\Users\\localadmin\\Downloads\\kuva.jpg", scale=1)
    fig.show()
    
    return 


def searchOpt(X,Y,Z,RMS):
        # X = feedrate
        # Y = dressing overlap ratio
        # Z = rotational velocity (rpm)
        # cylinder diameter (mm)
        d = 29
        x = X.flatten()
        y = Y.flatten()
        z = Z.flatten()
        # depth of cut
        ae = x/z
        # workpiece peripheral velocity 
        v_w = z/60*np.pi*d
        # specific material removal rate
        Qw = ae*v_w
        rms = RMS.flatten()
        print(Qw)
        
        return 

def main():

    data = pd.read_csv('toisenasteenanova.csv')
    y = data[['Aachen']].values
    X2 = data[['x1', 'x2', 'x3', 'x1x2', 'x1x3', 'x2x3', 'x1x2x3', 'x1x1', 'x2x2', 'x3x3', 'x1x1x2', 'x1x1x3', 'x2x2x1', 'x2x2x3', 'x3x3x1', 'x3x3x2', 'x1x1x2x2', 'x1x1x3x3', 'x2x2x3x3', 'x1x1x2x2x3x3']].values
    #y = data[['P']].values
    #X2 = data[['x1', 'x2', 'x3', 'x1x2', 'x1x3', 'x2x3', 'x1x2x3', 'x1x1', 'x2x2', 'x3x3']].values
    # Alla oleva on tärkeä rivi
    X2 = np.insert(X2, 0, np.ones(len(X2)), axis=1)
    yhat, SSE, C, b2, sSquared, r = multivariateRegr(X2, y)
    print("b2 shape: ", b2.shape)
    print("")
    dime = len(b2)
    print(len(b2))
    #Avaimet = data[['x1', 'x2', 'x3', 'x1x2', 'x1x3', 'x2x3', 'x1x2x3', 'x1x1', 'x2x2', 'x3x3', 'x1x1x2', 'x1x1x3', 'x2x2x1', 'x2x2x3', 'x3x3x1', 'x3x3x2', 'x1x1x2x2', 'x1x1x3x3', 'x2x2x3x3', 'x1x1x2x2x3x3']].keys()
    Avaimet = data[['x1', 'x2', 'x3', 'x1x2', 'x1x3', 'x2x3', 'x1x2x3', 'x1x1', 'x2x2', 'x3x3']].keys()
    anolla = pd.Index(['a0'])
    Avaimet = Avaimet.append(anolla)
    asList = Avaimet.tolist()
    asList[len(Avaimet)-1] = 'x1'
    asList[0] = 'a0'
    
    dime2 = len(yhat)
    fitSize = X2.shape[1]
    modX2 = X2
    CoD = np.zeros([1, fitSize])
    removedCoefs = []
    j = 0
    while j < fitSize:

        yhat, SSE, C, b2, sSquared, r = multivariateRegr(modX2, y)
        dime = len(b2)
        dime2 = len(yhat)
        F, p = anova(C, sSquared, dime, b2)

        CoD[0,j] = coefOfDeterm(yhat, SSE, dime2)
        max_index = np.argmax(p, axis=0)
        delIndex = max_index
        
       #print(p)
        #print(max_index)
        #print("ModX2: ")
        #print(modX2)
        #print("Pmax: ", p[delIndex])
        if p[max_index] > 0.05:
            try:
                modX2 = np.delete(modX2, delIndex, 1)
                removedCoefs.append(asList[delIndex])
                asList.remove(asList[delIndex])
            except IndexError:
                #print("p: ",p)
                #print(len(p))
                print("Index error!")
                
            # Joko asList muutetaan takaisin pandan indexiksi tai sitten
            # Avaimet.drop pitää muuttaa, että poistetaankin asList listasta
            #print(asList)
            
            #Avaimet = Avaimet.drop(Avaimet[max_index])
            
            # Muista manuaalisesti testata tulos 
            
        else:
            print("loop finished!")
            break
        j = j+1
      
    print("b2: ", b2)
    print("")
    print("yhat: ", yhat)
    print("")
    print("F-values: ", F)
    print("")
    print("p-values: ", p*100)
    print("")
    
    # Removed coefs printtaa väärät kertoimet
    #print("Removed Coefs: ", removedCoefs)
    print("R2: ", CoD)
    print("Finished!")
    #print(modX2)
    plottaus(b2, opt=True)
    #searchOpt(np.array([1.5]), np.array([8.6]), np.array([100]), np.array([91.4]))
    
main()
