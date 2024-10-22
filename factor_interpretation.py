import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def handle(ser):
    result = ser.str.extract(pat='(\d+\.?\d*)')
    return result
    
def FeatureImportance(estimator,data_etr:pd.DataFrame):
    importance = pd.Series(estimator.feature_importances_, index=data_etr.columns)
    importance.sort_values().plot(kind='barh')
    plt.show()
    return importance

dfx=pd.read_csv("D:\\Data\\NewData.csv",encoding='utf-8')
df=dfx.copy()
df.dropna(axis=0,subset =df.columns.tolist(),inplace=True) 
dfx2 = df.copy()
df.drop(['Time'],axis=1,inplace=True)
lis=df.columns.tolist()
df=df.reset_index(drop=True)
df=df.astype(str)
data=df.iloc[:,0]
for i in range(1,16):
    data=pd.concat([data,handle(df.iloc[:,i])],axis=1)
data.columns=lis
data=data.astype('float64')
etr_lis = ['Temp', 'pH', 'Conductivity', 'NH4', 'TP','Do']
data_etr = data.loc[:,etr_lis]
dfx2['Time'] = pd.to_datetime(dfx2['Time'])
dfx2.set_index('Time',inplace = True)

data_vis = data_etr.copy()
scaler = MinMaxScaler()
data_vis[['Temp','Do','pH']] = scaler.fit_transform(data_vis[['Temp','Do','pH']])
#x = np.linspace(0,len(data_vis),len(data_vis)) # 创建x的取值范围
x = dfx2.index
plt.figure(1)
plt.plot(x, data_vis['Do'], label='Do')
plt.plot(x, data_vis['Temp'], label='Temp')
plt.legend()
plt.figure(2)
plt.plot(x, data_vis['Do'], label='Do')
plt.plot(x, data_vis['pH'], label='pH')
plt.legend()
plt.show()

label=data_etr.pop('Do')
shap.initjs()
model = xgb.XGBRegressor()
model.fit(data_etr, label)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_etr)
shap.summary_plot(shap_values, data_etr)

shap.force_plot(explainer.expected_value, shap_values[10], data_etr.iloc[10])
xg = xgb.XGBRegressor()
xg.fit(data_etr,label)
FeatureImportance(xg,data_etr)
