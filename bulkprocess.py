import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
def sanitycheck(df):
	DF = df[df.index > 0]
	DF = DF[DF.index < 500]
	DF = DF.reindex(sorted(DF.columns),axis = 1)
	return DF

def provN(foldna):
	f21 = glob(foldna + '/channel 1/2019/sep/*00001.ddf')
	f22 = glob(foldna + '/channel 2/2019/sep/*00001.ddf')
	f23 = glob(foldna + '/channel 3/2019/sep/*00001.ddf')
	f24 = glob(foldna + '/channel 4/2019/sep/*00001.ddf')

	all1 = pd.DataFrame(index=pd.read_csv(f21[1],header=25,sep='\t')['length (m)'].values)
	for i in range(len(f21)):
	    fram = pd.read_csv(f21[i],header=25,sep='\t')
	    fram.set_axis(['depth','tempc','stokes','nonstokes'],axis=1,inplace=True)
	    all1[str(f21[i])[-25:-25+8]+str(f21[i])[-25+9:-25+15]]=fram.tempc.values
	p1 = sanitycheck(all1)
	print('Channel 1 is processed')

	all2 = pd.DataFrame(index=pd.read_csv(f22[1],header=25,sep='\t')['length (m)'].values)

	for i in range(len(f22)):
	    fram = pd.read_csv(f22[i],header=25,sep='\t')
	    fram.set_axis(['depth','tempc','stokes','nonstokes'],axis=1,inplace=True)
	    all2[str(f22[i])[-25:-25+8]+str(f22[i])[-25+9:-25+15]]=fram.tempc.values
	p2 = sanitycheck(all2)
	print('Channel 2 is processed')

	all3 = pd.DataFrame(index=pd.read_csv(f23[1],header=25,sep='\t')['length (m)'].values)
	for i in range(len(f23)):
	    fram = pd.read_csv(f23[i],header=25,sep='\t')
	    fram.set_axis(['depth','tempc','stokes','nonstokes'],axis=1,inplace=True)
	    all3[str(f23[i])[-25:-25+8]+str(f23[i])[-25+9:-25+15]]=fram.tempc.values
	p3 = sanitycheck(all3)
	print('Channel 3 is processed')

	all4 = pd.DataFrame(index=pd.read_csv(f23[1],header=25,sep='\t')['length (m)'].values)

	for i in range(len(f24)):
	    fram = pd.read_csv(f24[i],header=25,sep='\t')
	    fram.set_axis(['depth','tempc','stokes','nonstokes'],axis=1,inplace=True)
	    all4[str(f24[i])[-25:-25+8]+str(f24[i])[-25+9:-25+15]]=fram.tempc.values
	p4 = sanitycheck(all4)
	print('Channel 4 is processed')

	p1.to_csv('c1_data_sofar.csv')
	p2.to_csv('c2_data_sofar.csv')
	p3.to_csv('c3_data_sofar.csv')
	p4.to_csv('c4_data_sofar.csv')
	return p1,p2,p3,p4

def framepS(df,chan):
	plt.figure(figsize=(12,8))
	df.plot(legend=False,alpha=0.05,ax = plt.gca(),color = 'grey')
	df[df.columns[-1]].plot(legend=True,color = 'blue',ax = plt.gca(),label=df.columns[-1])
	df[df.columns[0]].plot(legend=True,color = 'red',ax = plt.gca(),label=df.columns[0])
	plt.ylim(-2,50)
	plt.xlabel('Distance(m)')
	plt.ylabel('Temperature ($\degree$C)')
	plt.title('Temperature measured with DTS in Channel'+str(chan))
	plt.savefig('ch'+str(chan)+'.png',dpi=300)
	print('Channel '+str(chan)+' is plotted and saved as file ch'+str(chan)+'.png')

def newplots():
	P1 = pd.read_csv('c1_data_sofar.csv',index_col=0)
	P2 = pd.read_csv('c2_data_sofar.csv',index_col=0)
	P3 = pd.read_csv('c3_data_sofar.csv',index_col=0)
	P4 = pd.read_csv('c4_data_sofar.csv',index_col=0)
	framepS(P1,1)
	framepS(P2,2)
	framepS(P3,3)
	framepS(P4,4)
	print('All plots are now updated.')