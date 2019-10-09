from __future__ import division
import numpy as np
import pandas as pd
import sympy as sp
from math import factorial as fac
def Djk(j,K):
    M=K/2
    m=int((j+1)/2)
    up=min(j,M)
    low=m
    if low == up:
        n = m
        R1= float(n**M*fac(2*n))
        R2= float(fac(M-n)*fac(n)*fac(n-1)*fac(j-n)*fac(2*n-j))
        return (R1/R2)*(-1)**(j+M) 
    else:
        R = 0
        for n in np.arange(low,up+1):
            R1= float(n**M*fac(2*n))
            R2= float(fac(M-n)*fac(n)*fac(n-1)*fac(j-n)*fac(2*n-j))
            R += R1/R2
        return R *(-1)**(j+M)
vis=np.vectorize(Djk)(np.arange(1,15),14) 
def StehF(time,fun,args):
	if type(args) == type(1) or type(0.01):
		TD = 0.0
		A = np.log(2)/time
		for i in range(1,14+1):
			TD = TD+vis[i-1]*fun(args)
		TD = A*TD
		return TD
	elif len(args)== 2:
		A = np.log(2)/time
		TD = 0.0
		for i in range(1,14+1):
			TD = TD+vis[i-1]*fun(args[0],args[1])
		TD = A*TD
		return TD
# Refer to: Villinger, H., 1985, Solving cylindrical geothermal problems using
# Gaver-Stehfest inverse Laplace transform, Geophysics, vol. 50 no. 10 p.
# 1581-1587

from scipy.special import jv,kv,iv,yv
bessel = {'besselj': jv,'besselk':kv,'besseli':iv,'bessely':yv}
libraries = [bessel, "numpy"]
from mpmath import *
mp.dps = 15; mp.pretty = True

I0 = lambda x: float(besseli(0,x))
I1 = lambda x: float(besseli(1,x))
K0 = lambda x: float(besselk(0,x))
K1 = lambda x: float(besselk(1,x))

Hg = 1;kappa = 0.99
hk=lambda Hg,s,kappa: np.sqrt(Hg*s/kappa)

def B2B1(s,Ns,Ng,N12,Hg,Hf,kappa,AD1,AD2,rDb): 
    HK=hk(Hg,s,kappa)
    TOP=kappa*HK*I1(rDb*HK)*K0(rDb*np.sqrt(s))+np.sqrt(s)*I0(rDb*HK)*K1(rDb*np.sqrt(s))
    BOT=kappa*HK*K1(rDb*HK)*K0(rDb*np.sqrt(s))+(-1)*np.sqrt(s)*K0(rDb*HK)*K1(rDb*np.sqrt(s))
    return TOP/BOT

def B4B1(s,Ns,Ng,N12,Hg,Hf,kappa,AD1,AD2,rDb): 
    HK=hk(Hg,s,kappa)
    TOP=kappa*HK*I1(rDb*HK)*K0(rDb*HK)+kappa*HK*I0(rDb*HK)*K1(rDb*HK)
    BOT=kappa*HK*K1(rDb*HK)*K0(rDb*np.sqrt(s))+(-1)*np.sqrt(s)*K0(rDb*HK)*K1(rDb*np.sqrt(s))
    return TOP/BOT

RSS,zD,x = sp.symbols("RSS zD x")
Tin=4
def nonD(T,Tin):
	return (T-Trs)/(Tin-Trs)
def DIM(TD,Tin):
	return TD*(Tin-Trs)+Trs
def Rsft(t,ks,alphs,rb):
	rn= 1/4/np.pi/ks*np.log(4*alphs*t/1.78/rb**2)
	if rn > 0.035:
		return rn
	else:
		return 0.035

# R_profile = pd.read_csv('geo.csv',sep='\t',index_col=0) #Turns out to be more influential than was estimated. 
# coeff = np.polyfit(R_profile.index.values/max(R_profile.index.values),R_profile['3K'],2)
# tbx = lambda z: coeff[0]*z**2 + coeff[1]*z**1 + coeff[2]*z**0# + coeff[3]*z + coeff[4]
# Trs=max(R_profile['3K'].values)

class cbhe: #Flow direction? I'll just use this as the known... one way or the other... something doesn't make too much sense just yet. 
	def __init__(self,Len=408.43,ks=3.25,cs=2.241e6,thick=2.4e-3,w = 0.58e-3,Tin=4, Q = 6360,kg = 2.77, kep = 22.5,deo = 0.1651):
		# Len=185-17;Q=-3500
		self.ks=ks#W/mK
		self.Len=Len
		cs=2.241e6;
		self.alphs=ks/cs;
		# self.rb = 115e-3/2#Borehole radius
		self.rb = deo/2
		#w = 0.58e-3#water rate
		cf=4186*999.1 #J/m3/K
		#Innerpipe
		dpo=88.9e-3;#thick=2.4e-3;
		kpp=0.4#W/mK
		dpi=dpo-2*thick
		#Outerpipe
		dei=deo-2*(4e-3);#kep=0.4#W/mK
		AD1=(dpi/deo)**2;AD2=(dei**2-dpo**2)/deo**2
		Ns=2*np.pi*ks*Len/w/cf
		vpi=w/(np.pi/4*dpi**2)
		muf=1.138e-3;rhof=999.1;kf=0.589;Pr=8.09
		Re=rhof*vpi*dpi/muf
		f_church= lambda re: 2/(1/((8.0/re)**10+(re/36500.0)**20)**0.5+ (2.21*np.log(re/7.0))**10)**0.2
		NuGl= lambda re: f_church(re)/2.0*(re-1000)*Pr/(1+12.7*(f_church(re)/2.0)**0.5*(Pr**(2.0/3)-1) )
		def Nu(re):
		    Nulam=4.364
		    if re<2000:
		        return Nulam
		    else:
		        return NuGl(re)
		hpi=kf/dpi*Nu(Re)
		Rfpi=1/np.pi/dpi/hpi
		deq=dei-dpo
		Aan=np.pi/4*(dei**2-dpo**2)
		van=w/Aan
		Rean=rhof*van*deq/muf
		Nuan=Nu(Rean)
		hpo=kf/deq*Nu(Rean)
		Rfpo=1/np.pi/dpo/hpo
		Rpp=np.log(dpo/dpi)/2/np.pi/kpp
		R12=Rfpi+Rpp+Rfpo
		N12=Len/R12/w/cf
		Radd=np.log((deo+4e-3)/deo)/2/np.pi/kg
		Rg=1/np.pi/dei/hpo+1/2/np.pi/kep*np.log(deo/dei)+Radd#External pipe wall resistance is significantly smaller than pipe
		Ng=Len/w/cf/Rg
		kg=ks*0.99
		cg=cs
		kappa=0.99;Hg=1
		Hf=cf/cs
		rDb=165e-3/deo
		# self.Tin = Tin
		# Tdb=nonD(tbx(zD),self.Tin)
		# dTdb=sp.diff(Tdb)
		cw=4.19e6
		self.Ns = Ns; self.N12= N12;self.Ng = Ng;self.AD1 = AD1;self.AD2 = AD2;self.rDb = rDb;self.Hf = Hf;self.Q = Q;self.Rg = Rg

	def calS(self,s):#Calculates the resulting four elements of temperature at whatever timestep with the reference heat injection rate
		HK = hk(Hg,s,kappa)
		b2b1 = B2B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		C0 = 1 - 1/(1-kappa*self.Ns/self.Ng*HK*(I1(HK)-b2b1*K1(HK))/(I0(HK)+b2b1*K0(HK)))
		b4b1 = B4B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		beta = -self.Ns*self.Hf*self.AD2/2*s-self.Ng*C0+self.Ns*self.Hf/2*self.AD1*s
		gamma= -(self.Ns*self.Hf*self.AD2/2*s+self.N12+self.Ng*C0)*(self.Ns*self.Hf*self.AD1/2*s+self.N12)+self.N12*self.N12
		a1 = - beta/2+np.sqrt(beta**2-4*gamma)/2
		a2 = - beta/2 +(-1)*np.sqrt(beta**2-4*gamma)/2
		b1 = (a1+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12
		b2 = (a2+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12 #Check equation IS THIS RIGHT?
		C1 = self.Ns/s/(1-b1)/(1-exp(a1)/exp(a2))
		C2 = (self.Ns/s-(1-b1)*C1)/(1-b2)
		C3 = b1*C1
		C4 = b2*C2
		F1 = lambda zD: C1*exp(a1*zD)+C2*exp(a2*zD)
		F2 = lambda zD: C3*exp(a1*zD)+C4*exp(a2*zD)
		FS = lambda zD,rD: (F2(zD)*self.Ng/kappa/self.Ns*b4b1*K0(rD*np.sqrt(s)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		FG = lambda zD,rD: (F2(zD)*self.Ng/kappa/self.Ns*(I0(rD*HK)+b2b1*K0(rD*HK)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		# return HK,b2b1,C0,b4b1,beta,gamma,a1,a2,b1,b2,C1,C2,C3,C4,F1,F2,FS,FG
		self.F1,self.F2,self.FS,self.FG = F1,F2,FS,FG
		return F1,F2,FS,FG
	#Essentially everything above is a CHECK and so far makes sense... most sense. The rest of the problem is how to initiate a 14-element Fx array from a s
	def g4(self,time,zD):
		A = np.log(2)/time
		f1,f2,fs,fg = 0,0,0,0
		for i in range(1,15):
		    F1,F2,Fs,Fg = self.calS(i*A)
		    f1 += vis[i-1]*F1(zD)
		    f2 += vis[i-1]*F2(zD)
		    fs += vis[i-1]*Fs(zD,1)
		    fg += vis[i-1]*Fg(zD,1)
		f1 = f1*A;f2=f2*A;fs=fs*A;fg=fg*A
		return f1,f2,fs,fg

	def calSV(self,s,zD):#Calculates the resulting four elements of temperature at whatever timestep with the reference heat injection rate
		HK = hk(Hg,s,kappa)
		b2b1 = B2B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		C0 = 1 - 1/(1-kappa*self.Ns/self.Ng*HK*(I1(HK)-b2b1*K1(HK))/(I0(HK)+b2b1*K0(HK)))
		b4b1 = B4B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		beta = -self.Ns*self.Hf*self.AD2/2*s-self.Ng*C0+self.Ns*self.Hf/2*self.AD1*s
		gamma= -(self.Ns*self.Hf*self.AD2/2*s+self.N12+self.Ng*C0)*(self.Ns*self.Hf*self.AD1/2*s+self.N12)+self.N12*self.N12
		a1 = - beta/2+np.sqrt(beta**2-4*gamma)/2
		a2 = - beta/2 +(-1)*np.sqrt(beta**2-4*gamma)/2
		b1 = (a1+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12
		b2 = (a2+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12 #Check equation IS THIS RIGHT?
		C1 = self.Ns/s/(1-b1)/(1-exp(a1)/exp(a2))
		C2 = (self.Ns/s-(1-b1)*C1)/(1-b2)
		C3 = b1*C1
		C4 = b2*C2
		T1 = C1*exp(a1*zD)+C2*exp(a2*zD)
		T2 =  C3*exp(a1*zD)+C4*exp(a2*zD)
		TS = lambda rD: (T2*self.Ng/kappa/self.Ns*b4b1*K0(rD*np.sqrt(s)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		TG = lambda rD: (T2*self.Ng/kappa/self.Ns*(I0(rD*HK)+b2b1*K0(rD*HK)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		# return HK,b2b1,C0,b4b1,beta,gamma,a1,a2,b1,b2,C1,C2,C3,C4,T1,T2,TS,TG		
		return T1,T2,TS,TG		
	def DIM(self,tar):
		return 8.4+self.Q*tar/2/np.pi/ks/self.Len
	# def convttime(hour):
 #    	return ks*hour*3600/cs/(deo/2)**2


class eanu:
	def __init__(self,Len=185-17,ks=3.25,cs=2.241e6,rb=115e-3,thick=2.4e-3,w = 0.58e-3,Tin=4, Q = 6360):
		# Len=185-17;Q=-3500
		self.ks=ks#W/mK
		self.Len=Len
		cs=2.241e6;
		self.alphs=ks/cs;
		self.rb = 115e-3/2#Borehole radius
		#w = 0.58e-3#water rate
		cf=4186*999.1 #J/m3/K
		#Innerpipe
		dpo=40e-3;#thick=2.4e-3;
		kpp=0.4#W/mK
		dpi=dpo-2*thick
		#Outerpipe
		deo=115e-3;dei=deo-2*(0.4e-3);kep=0.4#W/mK
		# AD1=(dpi/deo)**2;AD2=(dei**2-dpo**2)/deo**2 #REVISE!#DONE!
		AD1 = (dei**2-dpo**2)/deo**2;AD2 = (dpi/deo)**2#DONE!
		Ns=2*np.pi*ks*Len/w/cf
		vpi=w/(np.pi/4*dpi**2)
		muf=1.138e-3;rhof=999.1;kf=0.589;Pr=8.09
		Re=rhof*vpi*dpi/muf
		f_church= lambda re: 2/(1/((8.0/re)**10+(re/36500.0)**20)**0.5+ (2.21*np.log(re/7.0))**10)**0.2
		NuGl= lambda re: f_church(re)/2.0*(re-1000)*Pr/(1+12.7*(f_church(re)/2.0)**0.5*(Pr**(2.0/3)-1) )
		def Nu(re):
		    Nulam=4.364
		    if re<2000:
		        return Nulam
		    else:
		        return NuGl(re)
		hpi=kf/dpi*Nu(Re)
		Rfpi=1/np.pi/dpi/hpi
		deq=dei-dpo
		Aan=np.pi/4*(dei**2-dpo**2)
		van=w/Aan
		Rean=rhof*van*deq/muf
		Nuan=Nu(Rean)
		hpo=kf/deq*Nu(Rean)
		Rfpo=1/np.pi/dpo/hpo
		Rpp=np.log(dpo/dpi)/2/np.pi/kpp
		R12=Rfpi+Rpp+Rfpo
		N12=Len/R12/w/cf
		Radd=0#Added grout
		Rg=1/np.pi/dei/hpo+1/2/np.pi/kep*np.log(deo/dei)+Radd#External pipe wall resistance is significantly smaller than pipe
		Ng=Len/w/cf/Rg
		kg=ks*0.99
		cg=cs
		kappa=0.99;Hg=1
		Hf=cf/cs
		rDb=165e-3/deo
		# self.Tin = Tin
		# Tdb=nonD(tbx(zD),self.Tin)
		# dTdb=sp.diff(Tdb)
		cw=4.19e6
		self.Ns = Ns; self.N12= N12;self.Ng = Ng;self.AD1 = AD1;self.AD2 = AD2;self.rDb = rDb;self.Hf = Hf;self.Q = Q

	def calS(self,s):#Calculates the resulting four elements of temperature at whatever timestep with the reference heat injection rate
		HK = hk(Hg,s,kappa)
		b2b1 = B2B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		C0 = 1 - 1/(1-kappa*self.Ns/self.Ng*HK*(I1(HK)-b2b1*K1(HK))/(I0(HK)+b2b1*K0(HK)))
		b4b1 = B4B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		# beta = -self.Ns*self.Hf*self.AD2/2*s-self.Ng*C0+self.Ns*self.Hf/2*self.AD1*s #REVISE!
		beta = -self.Ns*self.Hf*self.AD2/2*s +self.Ng*C0+self.Ns*self.Hf/2*self.AD1*s #REVISE!#DONE!
		# gamma= -(self.Ns*self.Hf*self.AD2/2*s+self.N12+self.Ng*C0)*(self.Ns*self.Hf*self.AD1/2*s+self.N12)+self.N12*self.N12 #REVISE!
		gamma= -(self.Ns*self.Hf*self.AD1/2*s+self.N12+self.Ng*C0)*(self.Ns*self.Hf*self.AD2/2*s+self.N12)+self.N12*self.N12 #REVISE!

		a1 = - beta/2+np.sqrt(beta**2-4*gamma)/2
		a2 = - beta/2 +(-1)*np.sqrt(beta**2-4*gamma)/2
		b1 = (a1+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12 #REVISE! #DONE!
		b2 = (a2+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12 #REVISE! #DONE!
		C3 =  -self.Ns/s/(1-b1)/(1-exp(a1)/exp(a2))#REVISE!
		C4 = (-self.Ns/s-(1-b1)*C3)/(1-b2)#REVISE! #DONE!
		C1 = b1*C3 #REVISE!
		C2 = b2*C4 #REVISE!
		F1 = lambda zD: C1*exp(a1*zD)+C2*exp(a2*zD)
		F2 = lambda zD: C3*exp(a1*zD)+C4*exp(a2*zD)
		FS = lambda zD,rD: (F1(zD)*self.Ng/kappa/self.Ns*b4b1*K0(rD*np.sqrt(s)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		FG = lambda zD,rD: (F1(zD)*self.Ng/kappa/self.Ns*(I0(rD*HK)+b2b1*K0(rD*HK)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		# return HK,b2b1,C0,b4b1,beta,gamma,a1,a2,b1,b2,C1,C2,C3,C4,F1,F2,FS,FG
		self.F1,self.F2,self.FS,self.FG = F1,F2,FS,FG
		return F1,F2,FS,FG
	#Essentially everything above is a CHECK and so far makes sense... most sense. The rest of the problem is how to initiate a 14-element Fx array from a s
	def g4(self,time,zD):
		A = np.log(2)/time
		f1,f2,fs,fg = 0,0,0,0
		for i in range(1,15):
		    F1,F2,Fs,Fg = self.calS(i*A)
		    f1 += vis[i-1]*F1(zD)
		    f2 += vis[i-1]*F2(zD)
		    fs += vis[i-1]*Fs(zD,1)
		    fg += vis[i-1]*Fg(zD,1)
		f1 = f1*A;f2=f2*A;fs=fs*A;fg=fg*A
		return f1,f2,fs,fg

	def calSV(self,s,zD):#Calculates the resulting four elements of temperature at whatever timestep with the reference heat injection rate
		HK = hk(Hg,s,kappa)
		b2b1 = B2B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		C0 = 1 - 1/(1-kappa*self.Ns/self.Ng*HK*(I1(HK)-b2b1*K1(HK))/(I0(HK)+b2b1*K0(HK)))
		b4b1 = B4B1(s,self.Ns,self.Ng,self.N12,Hg,self.Hf,kappa,self.AD1,self.AD2,self.rDb)
		beta = -self.Ns*self.Hf*self.AD2/2*s-self.Ng*C0+self.Ns*self.Hf/2*self.AD1*s
		gamma= -(self.Ns*self.Hf*self.AD2/2*s+self.N12+self.Ng*C0)*(self.Ns*self.Hf*self.AD1/2*s+self.N12)+self.N12*self.N12
		a1 = - beta/2+np.sqrt(beta**2-4*gamma)/2
		a2 = - beta/2 +(-1)*np.sqrt(beta**2-4*gamma)/2
		b1 = (a1+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12
		b2 = (a2+self.Ns*self.Hf/2*self.AD1*s+self.N12)/self.N12 #Check equation IS THIS RIGHT?
		C1 = self.Ns/s/(1-b1)/(1-exp(a1)/exp(a2))
		C2 = (self.Ns/s-(1-b1)*C1)/(1-b2)
		C3 = b1*C1
		C4 = b2*C2
		T1 = C1*exp(a1*zD)+C2*exp(a2*zD)
		T2 =  C3*exp(a1*zD)+C4*exp(a2*zD)
		TS = lambda rD: (T2*self.Ng/kappa/self.Ns*b4b1*K0(rD*np.sqrt(s)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		TG = lambda rD: (T2*self.Ng/kappa/self.Ns*(I0(rD*HK)+b2b1*K0(rD*HK)))/(self.Ng/kappa/self.Ns*(I0(HK)+b2b1*K0(HK))+(-1)*HK*I1(HK)+b2b1*HK*K1(HK))
		# return HK,b2b1,C0,b4b1,beta,gamma,a1,a2,b1,b2,C1,C2,C3,C4,T1,T2,TS,TG		
		return T1,T2,TS,TG		
	def DIM(self,tar):
		return 8.4+self.Q*tar/2/np.pi/ks/self.Len
	# def convttime(hour):
 #    	return ks*hour*3600/cs/(deo/2)**2
