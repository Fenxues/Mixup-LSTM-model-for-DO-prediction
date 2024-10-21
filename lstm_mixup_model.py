import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
def saveColumn(data:pd.DataFrame,cols:list):
    df = pd.DataFrame()
    for col in cols:
      df=pd.concat([df,data[col]],axis=1)
    return df

def inner(x,val):
            if str(x)=='0Z'or str(x)=='10011.90T':
                x =val
                return x
            return x

def filNa(*wargs):
    df=pd.DataFrame()
    if len(wargs)==3:
      i=0
      for col in wargs[2]:

         wargs[0][col]=wargs[0][col].apply(inner,args=(wargs[1][i],))
         wargs[0][col]=  wargs[0][col].apply(lambda x: float(x))
         wargs[0][col]=wargs[0][col].fillna(wargs[0][col].mean())
         i+=1

    elif len(wargs)==1:
        for col in wargs.columns.tolist():
            wargs[0][col] = wargs[0][col].fillna(wargs[0][col].mean())

    df=wargs[0]
    return df

def scale(train, test):

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    train_scaled = scaler.transform(train)
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

def missdata_situation(X):
     print(X.isnull().sum(),'\n')  
     total = X.isnull().sum().sort_values(ascending=False)
     percent = (X.isnull().sum()/X.isnull().count()).sort_values(ascending=False)
     missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
     print(missing_data,'\n')

def timeRh(df):
    for i in range(4,df.shape[0]):
        lam =0.5
        df.iloc[i,:]=lam*df.iloc[i-1,:]+lam*df.iloc[i,:]
    return df

def createXY(dataset,n_past,predict_column,out_time_steps):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i:i+out_time_steps,predict_column])
            if i==len(dataset)-out_time_steps:
                break;
    return np.array(dataX),np.array(dataY)

def build_model(optimizer = 'adam',out_time_steps = 12):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(input_time_steps,features)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(out_time_steps))
    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

def Mypre(X,Y,target_column,model,scaler):
    """
    @parameter:
    X:(1,in_time_steps,features)
    Y:(1,out_time_steps)
    @return:
    orgin:(out_times_steps,)
    pred:(out_times_steps,)
    """
    prediction=model.predict(X)
   
    #print(prediction.shape)
    prediction=np.reshape(prediction,(prediction.shape[1],prediction.shape[0]))
    #print(prediction.shape)
    prediction_copies_array = np.repeat(prediction,X.shape[2], axis=1)
    #print(prediction_copies_array.shape)
  
    pred=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),X.shape[2])))[:,target_column]
   
    original=np.reshape(Y,(Y.shape[1],Y.shape[0]))
    #print(original.shape)
    original_copies_array = np.repeat(original,X.shape[2], axis=1)
    #print(original_copies_array.shape)
    
    original=scaler.inverse_transform(np.reshape(original_copies_array,(len(original),X.shape[2])))[:,target_column]
    return original,pred

def appearance_test(preX, preY, predict_column, model, scaler):
    original=[]
    prediction=[]
    for i in range(preX.shape[0]):
        origin,pred=Mypre(preX[i][np.newaxis,:],preY[i][np.newaxis,:],predict_column,model,scaler)
        original.append(origin)
        prediction.append(pred)
    return original,prediction

def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def comment(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    import math

    MAE = mean_absolute_error(y_true, y_pred)
    MSE = mean_squared_error(y_true, y_pred)
    R2=r2_score(y_true, y_pred)
    RMSE = math.sqrt(MSE)
    print("Root Mean Square Error:",RMSE)
    print("Mean Square Error:",MSE)
    print("Mean Absolute Error:",MAE)
    print('Mean absolute percentage error:',mape(y_true, y_pred))
    print('R-squared:',R2)
df=pd.read_csv("D://Data//NEWDF.csv",parse_dates=["Time"],index_col=[1])
df.drop('Unnamed: 0',axis=1,inplace=True)test_split=round(len(df)*0.20)
test_split= 2091
df_for_training=df[:-test_split]
df_for_testing=df[-test_split:]
spilt_for_predict=test_split
df_for_prediction_ori=df[-spilt_for_predict:]
df_for_training=timeRh(df_for_training)
df_for_testing=timeRh(df_for_testing)
df_for_prediction=timeRh(df_for_prediction_ori)

scaler1,df_for_training_scaled,df_for_testing_scaled=scale(df_for_training,df_for_testing)
scaler2,df_for_training_scaled,df_for_prediction_scaled=scale(df_for_training,df_for_prediction)

input_time_steps=60
features=df.shape[1]
out_time_steps=12
predict_column=4

trainX,trainY=createXY(df_for_training_scaled,input_time_steps,predict_column,out_time_steps)
testX,testY=createXY(df_for_testing_scaled,input_time_steps,predict_column,out_time_steps)
preX,preY=createXY(df_for_prediction_scaled,input_time_steps,predict_column,out_time_steps)
cesas_model = build_model('adam',out_time_steps)
grid_model = KerasRegressor(build_fn=build_model)
model=grid_model.fit(trainX,trainY,validation_data=(testX,testY))
ori, pre = appearance_test(preX, preY, 4, model, scaler1)
original=np.array(ori)[:,0]
prediction=np.array(pre)[:,0]
comment(original,prediction)