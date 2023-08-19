#!/usr/bin/env python
# coding: utf-8

# In[8]:


import sympy as sp


#takes in a second-rank tensor (input as a an array), converts it to a matrix, calculates the inverse and returns it as a second-rank tensor
def Inverse(gll):  
    gllMat=sp.Matrix(gll)
    guu=sp.simplify(gllMat.inv())
    return sp.Array(guu)

#returns the square of an infinitesimal line segment. Note that Greek symbols like d(theta) will not be printed properly 
def LineltFromMetric(gll,Xu):
    Dim=len(Xu)
    dX_str=[]
    for i in range(Dim):  #returns list of infinitesimal coordinates as strings
        dX_str.append('d'+str(Xu[i]))  
    dX=[sp.symbols(v) for v in dX_str] #converts infinitesimal coordinates from strings to sympy variables. 
    dXsq=sp.tensorproduct(dX,dX) #constructs second-rank tensor to be contracted with the metric to get ds
    ds=sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(gll,dXsq),(0,2)),(0,1))
    return ds

#reconstructs metric from the square of a line element
def MetricFromDS(ds,Xu):
    Dim=len(Xu)
    gll=[[0 for i in range(Dim)] for j in range(Dim)] #creates empty second-rank tensor to be filled with metric elements
    dX_str=[]
    for i in range(Dim):  
        dX_str.append('d'+str(Xu[i]))  
        dX=[sp.symbols(v) for v in dX_str]
    dsi=sp.expand(ds)
    for i in range(Dim):
        gll[i][i]=sp.simplify(dsi.coeff(dX[i],2))  #diagonal elements of the metric 
        for j in range(i+1,Dim):
            gll[i][j]=sp.nsimplify(sp.S(1)/2*dsi.coeff(dX[i]*dX[j],1)) #all other elements of the metric, usually zero
            gll[j][i]=gll[i][j]
    return sp.Array(gll)    #returns gll as a second-rank tensor 


#Christoffel Connection of the first kind (low-low-low)
def GetCClll(gll,X):  
    Dim=len(X)
    CClllout=[[[0 for b in range(Dim)] for u in range(Dim)] for v in range(Dim)]
    for b in range(Dim):
        for u in range(Dim):
            for v in range(Dim):
                CClllout[b][u][v]=sp.nsimplify(sp.S(1)/2*(sp.diff(gll[b][u],X[v])+sp.diff(gll[b][v],X[u])-sp.diff(gll[u][v],X[b])))
    return sp.Array(CClllout)

#Christoffel Connection of the second kind (up-low-low)
def GetCCull(gll,X):  
    Dim=len(X)
    guu=Inverse(gll)
    CClllin=GetCClll(gll,X)
    CCull=sp.tensorcontraction(sp.tensorproduct(guu, CClllin),(0,2)) #contraction of first and third indices of the fifth-rank tensor product g^{\alpha\rho} \Gamma_{\sigma\mu\nu} 
    return CCull
                           
#Riemann tensor in up-low-low-low form
def GetRiemann(gll,X): 
    Dim=len(X)
    CCull=GetCCull(gll,X)
    Rulllout=[[[[0 for a in range(Dim)] for p in range(Dim)] for g in range(Dim)] for b in range(Dim)] #empty array with four indices
    for a in range(Dim):
        for p in range(Dim):
            for g in range(Dim):
                for b in range(Dim):
                    Rulllout[a][p][g][b]=sp.simplify(sp.diff(CCull[a][b][p],X[g])-sp.diff(CCull[a][g][p],X[b])+sum(CCull[a][g][s]*CCull[s][b][p] for s in range(Dim))-sum(CCull[a][b][s]*CCull[s][g][p] for s in range(Dim)))
    return sp.Array(Rulllout)

#Ricci tensor in low-low form
def GetRicci(gll,X): 
    Rulll=GetRiemann(gll, X)
    Rll=sp.tensorcontraction(Rulll,(0,2))
    return sp.trigsimp(sp.factor(Rll))

#Ricci scalar 
def GetRicciS(gll,X): 
    Rll=GetRicci(gll,X)
    guu=Inverse(gll)
    Rs=sp.tensorcontraction(sp.tensorcontraction(sp.tensorproduct(Rll,guu),(0,2)),(0,1))
    return sp.nsimplify(Rs)

#Returns Kerr metric with Kerr parameters. Note that Python is not equipped to calculate the Riemann tensor, the Ricci tensor, nor the Ricci scalar 
def GetKerrgll(M,a,r,T):
    kerrgll=sp.Array([[-1+2*M*r/(a**2*sp.cos(T)**2+r**2),0,0,-2*a*M*r*sp.sin(T)**2/(r**2+a**2*sp.cos(T)**2)],
    [0, (a**2*sp.cos(T)**2+r**2)/(a**2-2*M*r+r**2),0,0],
    [0,0,a**2*sp.cos(T)**2+r**2,0],
    [-2*a*M*r*sp.sin(T)**2/(r**2+a**2*sp.cos(T)**2),0,0,(sp.sin(T)**2*((a**2+r**2)**2-a**2*(a**2-2*M*r+r**2)*sp.sin(T)**2))/(a**2*sp.cos(T)**2+r**2)]])    
    return sp.nsimplify(kerrgll)




# In[ ]:





# In[ ]:




