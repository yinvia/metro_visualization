
# coding: utf-8

# In[1]:


#导入需要用到的库
import os,sys,pickle
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import matplotlib
from dateutil.parser import parse
from pyecharts import Map
from pyecharts import Geo
from mpl_toolkits.basemap import Basemap
import warnings
warnings.filterwarnings("ignore")
import folium
from folium.plugins import HeatMap
import re


# In[2]:


#日期与时间
import time
from datetime import datetime
from datetime import date


# In[3]:


#？导入训练集和测试集？
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score, auc, roc_curve
from sklearn.preprocessing import MinMaxScaler


# In[4]:


#作图用
# display for this notebook
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# In[5]:


#导入数据
with open('C:/Users/ASUS/Desktop/o2odata/subway_data.csv','r',encoding="utf-8") as a:
    subway=pd.read_csv(a)


# In[6]:


#导入地铁站经纬度数据
with open(r'C:\Users\ASUS\Desktop\o2odata\python\data_poi_shen-zhen-shi-di-tie-xian-lu.csv',encoding="utf-8") as b:
    jingwei=pd.read_csv(b)
jingwei


# In[7]:


#改格式
subway.drop(['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1','close_date'],axis=1,inplace=True)
subway.drop(0, inplace = True)
subway.dropna()
metro=subway


# In[8]:


#时间切片
metro['deal_date'] = pd.to_datetime(metro['deal_date'])
metro['date_min_interval']=metro['deal_date'].dt.minute//10
metro['date_hour']=metro['deal_date'].dt.hour
metro['period']=metro['date_hour']*6+metro['date_min_interval']
metro


# In[9]:


#删除无效数据
metro=metro[metro['company_name'].str.contains('地铁')]
metro=metro[(metro['period']>=36)&(metro['period']<=68)]
metro=metro[metro.station.notnull()]


# In[10]:


metro=metro[~metro['station'].str.contains('1')]
metro=metro[~metro['station'].str.contains('2')]
metro=metro[~metro['station'].str.contains('3')]
metro=metro[~metro['station'].str.contains('4')]
metro=metro[~metro['station'].str.contains('5')]
metro=metro[~metro['station'].str.contains('6')]
metro=metro[~metro['station'].str.contains('7')]
metro=metro[~metro['station'].str.contains('8')]
metro=metro[~metro['station'].str.contains('9')]
metro=metro[~metro['station'].str.contains('0')]
metro=metro[~metro['station'].str.contains('归属未知')]


# In[11]:


metro['station']=metro['station'].str.replace('站','')
metro


# In[12]:


#提取地铁入站数据
metro_in=metro[metro['deal_type']=='地铁入站']
company_in=metro_in['company_name'].unique()
deal_type_in=metro_in['deal_type'].unique()
y_in=metro_in.groupby(by='company_name')['card_no'].count()
#提取地铁出站数据
metro_out=metro[metro['deal_type']=='地铁出站']
company_out=metro_out['company_name'].unique()
deal_type_out=metro_out['deal_type'].unique()
y_out=metro_out.groupby(by='company_name')['card_no'].count()

# 使用fivethirtyeight这个风格
plt.style.use("fivethirtyeight")
plt.barh(company_in,y_in,color="#87CEFA",label='入站')
plt.barh(company_out,-y_out,color='orange',label='出站')
plt.xlabel('人数')
plt.title('各地铁线路出入站人数')
plt.show()


# In[13]:


#入站比例饼图
plt.axes(aspect='equal')
plt.pie(y_in,labels=company_in,shadow=True,autopct='%1.2f%%')


# In[14]:


#出站比例饼图
plt.axes(aspect='equal')
plt.pie(y_out,labels=company_out,shadow=True,autopct='%1.2f%%')


# In[15]:


y2=metro_in.groupby('period')['card_no'].describe()['count'].to_frame('Ridership')
y2


# In[16]:


#总入站人数时间分布图
plt.plot(y2)


# In[17]:


#分线路入站人数时间分布图
y3=metro.groupby(['company_name','period'])['card_no'].describe()['count'].to_frame('Ridership')
y3.reset_index(inplace = True)
y3


# In[18]:


l1=y3[y3.company_name=='地铁一号线']
l2=y3[y3.company_name=='地铁二号线']
l3=y3[y3.company_name=='地铁三号线']
l4=y3[y3.company_name=='地铁四号线']
l5=y3[y3.company_name=='地铁五号线']
l7=y3[y3.company_name=='地铁七号线']
l9=y3[y3.company_name=='地铁九号线']
l11=y3[y3.company_name=='地铁十一号线']
x1=l1.period
x2=l2.period
x3=l1.period
x4=l4.period
x5=l5.period
x7=l7.period
x9=l9.period
x11=l11.period
z1=l1.Ridership
z2=l2.Ridership
z3=l3.Ridership
z4=l4.Ridership
z5=l5.Ridership
z7=l7.Ridership
z9=l9.Ridership
z11=l11.Ridership
plt.plot(x1,z1)
plt.plot(x2,z2)
plt.plot(x3,z3)
plt.plot(x4,z4)
plt.plot(x5,z5)
plt.plot(x7,z7)
plt.plot(x9,z9)
plt.plot(x11,z11)
plt.legend(['line1','line2','line3','line4','line5','line7','line9','line11'],loc='upper left')
plt.show()


# In[19]:


y4=metro_in.groupby('station')['card_no'].describe()['count'].to_frame('Ridership')
y5=metro_in.groupby(['station','period'])['card_no'].describe()['count'].to_frame('Ridership')


# In[20]:


y4


# In[21]:


y5


# In[22]:


jingwei.head()


# In[23]:


y6=y4.join(jingwei.set_index('station'),on='station')

y6.dropna(inplace=True)


# In[24]:


ruzhan=np.array(y6['Ridership'],dtype=np.float)
lat1=np.array(y6.lat[0:len(y6.lat)])
lon1=np.array(y6.lon[0:len(y6.lon)])
data2=[[lat1[i],lon1[i],ruzhan[i]] for i in range(len(lat1))]
data2


# In[60]:


m1 = folium.Map([22.5,114.], tiles='http://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}',attr='AutoNavi',zoom_start=11,control_scale = True)
m1.add_child(HeatMap(data2, radius=12, gradient={.4:'blue',.65:'yellow',1:'red'}))
m1.save(os.path.join(r'C:\Users\ASUS\Desktop\o2odata\python', 'Heatmap_1.html'))
m1


# In[26]:


y5=pd.DataFrame(y5)
y5


# In[27]:


#data3=[[lat1[i],lon1[i],ruzhan[i]] for i in range(len(lat1))]


# In[28]:


#m2 = folium.Map([22.5,114.], tiles='stamentoner', control_scale = True，zoom_start=11)
#HeatMap(data3).add_to(m2)
#m2.save(os.path.join(r'C:\Users\ASUS\Desktop\metro\python', 'Heatmap_进站.html'))
#m2


# In[29]:


#trans=subway[(subway['conn_mark']==1)&(subway['date_hour']<12)&(subway['date_hour']>=6)]
#bus=subway[(subway['date_hour']<12)&(subway['date_hour']>=6)]
#transfer=bus[bus["card_no"].isin(trans.card_no)]
#transfer=transfer[(transfer['deal_money']!=0)&(transfer['deal_type']=='地铁出站')]
#transfer['station']=transfer[~(transfer['station'].isin(1))]


# In[30]:


#transrider=transfer.groupby('station')['card_no'].describe()['count'].to_frame('transrider')
#transrider


# In[31]:


#x=pd.merge(jingwei,transrider,left_index=True, right_index=True,on='station',how='left')
#x


# In[32]:


#lon=np.array(jingwei['lon'][0:len(jingwei)])
#lat=np.array(jingwei['lat'][0:len(jingwei)])


# In[33]:


#transrider=transrider
#trans=np.array(transrider['transrider'],dtype=np.float)
#data1 = [[lat[i],lon[i],trans[i]] for i in range(len(jingwei))]


# In[34]:


#m = folium.Map([22.5,114.], tiles='stamentoner', zoom_start=11)
#HeatMap(data1).add_to(m)
#m.save(os.path.join(r'C:\Users\ASUS\Desktop\metro\python', 'Heatmap.html'))
#m.save('Heatmap_profit_zonghe.html')#存放路径记得改
#m

