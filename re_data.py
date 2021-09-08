import pandas as pd
import numpy as np

def re_data(data,nn=10,load=60,train_size=0.8):
  fdata=pd.DataFrame(np.array(data)[:,:-1],index=data.index)
  for i in range(1,load):
    for j in range(0,4):
      r=np.zeros(i)
      r=np.append(r,np.array(data.iloc[:-i,j]))
      fdata[str(i)+"A"+str(j)]=r
  x=fdata[load-1:]
  yy=np.array(data.Close[nn:])/np.array(data.Close[:-nn])
  yyy=np.append(yy,np.array([0]*nn))
  y=pd.Series(yyy[load-1:],index=x.index)
  x=x.div(np.array(x.iloc[:,0]),axis=0)
  x=x.iloc[:,1:]
  zz=x
  x=x.iloc[:-nn]
  y=y.iloc[:-nn]
  xx=x
  n=len(xx)
  i=int(train_size*n)
  x_train=xx.iloc[:i]
  x_test=xx.iloc[i:]
  y_train=y.iloc[:i]
  y_test=y.iloc[i:]
  return([x_train,x_test,y_train,y_test,zz])
