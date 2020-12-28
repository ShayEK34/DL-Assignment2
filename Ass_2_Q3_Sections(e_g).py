from tensorflow.keras.layers import Input, Embedding, add,Flatten, concatenate,Dropout, Dense,BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import *
from sklearn.metrics import mean_squared_error
from tensorflow.keras.optimizers import *
from tensorflow.keras.metrics import mean_squared_error,RootMeanSquaredError
from sklearn.model_selection import train_test_split
import datetime as dt
from catboost import CatBoostRegressor
from os import path
import numpy as np
import pandas as pd
import os.path
import os
import gc
import matplotlib.pyplot as plt


# feature extraction
# create holidays features
from tensorflow.python.keras.regularizers import l1


def create_holiday_features(holiday_path,data_path,output_path,year,flag=False):
    holiday_df=pd.read_csv(holiday_path)
    holiday_dict={}
    if flag == False:
        for index,row in holiday_df.iterrows():
            date = dt.datetime.strptime(row['date'],'%Y-%m-%d')
            if date.year==year:
                holiday_dict[date]=row
    else:
        for index, row in holiday_df.iterrows():
            date = dt.datetime.strptime(row['date'], '%Y-%m-%d')
            if date.year in year:
                holiday_dict[date] = row
    chunksize = 10 ** 6
    for chunk in pd.read_csv(data_path, chunksize=chunksize):
        new_data = []
        for index,row in chunk.iterrows():
            if len (row['date'].split())>1:
                split=row['date'].split()
                row['date']=split[0]
            date = dt.datetime.strptime(row['date'], '%Y-%m-%d')
            is_before_holiday_week=False
            is_after_holiday_week=False
            is_holiday_day=False
            holiday_type='none'
            holiday_locale='none'
            holiday_locale_name='none'
            transferred=False
            for holiday in holiday_dict.keys():
                if holiday==date:
                    holiday_det=holiday_dict.get(holiday)
                    if holiday_det['transferred']==True:
                        continue
                    is_holiday_day = True
                    holiday_type=holiday_det['type']
                    holiday_locale=holiday_det['locale']
                    holiday_locale_name=holiday_det['locale_name']
                if abs(holiday-date).days < 4 :
                    if holiday  < date:
                        is_before_holiday_week=True
                    else:
                        is_after_holiday_week=True
            payment_week=False
            if date.day < 4 or (date.day > 15 and date.day < 19):
                payment_week=True
            new_data.append([row['id'],row['date'],row['day'],row['month'],row['year'],row['weekday'],row['store_nbr'],row['item_nbr'],row['onpromotion'],is_before_holiday_week,is_after_holiday_week,is_holiday_day,holiday_type,holiday_locale,holiday_locale_name,payment_week])
        write_chunk_to_csv(new_data,output_path)

# extract transactions features
def create_transactions(path_to_data):
    transaction_df=pd.read_csv(path_to_data)
    transaction_df['date'] = pd.to_datetime(transaction_df['date'])
    transaction_df['date'] = transaction_df['date'].apply(lambda x: x.date())
    transaction_df['weekday'] = transaction_df['date'].apply(lambda x: x.weekday())
    transaction_df['day'] = transaction_df['date'].apply(lambda x: x.day)
    transaction_df['month'] = transaction_df['date'].apply(lambda x: x.month)
    transaction_df['year'] = transaction_df['date'].apply(lambda x: x.year)
    group_by_store_weekday=transaction_df.groupby(['store_nbr','weekday','month','year']).agg({'transactions':['mean','median','std']})
    group_by_store_weekday.to_csv(r'C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\data_features\store_transaction_day.csv')
    group_by_store_month = transaction_df.groupby(['store_nbr','month', 'year']).agg({'transactions': ['mean', 'median', 'std']})
    group_by_store_month.to_csv(r'C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\data_features\store_transaction_month.csv')
    weeks=[]
    for date in transaction_df['day'].items():
        if date[1] <=8:
            weeks.append(1)
        elif date[1] <= 15:
            weeks.append(2)
        elif date[1]<=22:
            weeks.append(3)
        else:
            weeks.append(4)
    transaction_df['session'] = weeks
    group_by_store_week = transaction_df.groupby(['store_nbr', 'session', 'month', 'year']).agg({'transactions': ['mean', 'median', 'std']})
    group_by_store_week.to_csv(r'C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\data_features\store_transaction_week.csv')

# data generator in order to preform incremntal learning beacuse the size of the data
def generate_data_from_file(path_to_data, batchsize):
    target = None
    while True:
        for batch in pd.read_csv(path_to_data,chunksize=batchsize):
            batch = batch.loc[batch['unit_sales'] >= 0]
            target = batch['unit_sales']
            batch.drop(columns=['unit_sales','id'], inplace=True)
            W_data = batch['perishable'].map({0: 1.0, 1: 1.25})
            target_normalize = np.log1p(target)
            columns = batch.columns
            batch[columns] = batch[columns].fillna(batch[columns].mode().iloc[0])
            yield [np.array(batch['day']),np.array(batch['month']),np.array(batch['year']),np.array(batch['weekday']),
                   np.array(batch['store_nbr']),np.array(batch['item_nbr']),np.array(batch['onpromotion']),
                   np.array(batch['is_before_holiday_week']),np.array(batch['is_after_holiday_week']),np.array(batch['is_holiday_day']),
                   np.array(batch['holiday_type']),np.array(batch['holiday_locale']),np.array(batch['holiday_locale_name']),
                   np.array(batch['payment']), np.array(batch['item_family']),np.array(batch['perishable']),
                   np.array(batch['city']),np.array(batch['state']),np.array(batch['type']),np.array(batch['cluster']),
                   np.array(batch['month_transactions_mean_last_year']), np.array(batch['month_transactions_median_last_year']),
                   np.array(batch['month_transactions_std_last_year']),np.array(batch['month_transactions_mean_last_month']),
                   np.array(batch['month_transactions_median_last_month']), np.array(batch['month_transactions_std_last_month']),
                   np.array(batch['mean_month']),np.array(batch['median_month']),np.array(batch['std_month']),np.array(batch['mean_year']),
                   np.array(batch['median_year']), np.array(batch['std_year'])], np.array(target_normalize),W_data.values
            target = None
            W_data=None
            gc.collect()

# model call backs 
def set_callbacks(description='run1',patience=10):
    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=10)
    cb = [cp,es,rlop]
    return cb

# we use this function as the model metrics, its rmse weighted
def NWRMSLE(y, pred, w):
    return mean_squared_error(y, pred, sample_weight=w)**0.5

# append data to csv file
def write_chunk_to_csv(new_data,output_path):
    if(path.exists(output_path)):
        df=pd.DataFrame(new_data)
        df.to_csv(output_path,mode='a',header=False,index=False)
    else:
        df = pd.DataFrame(new_data)
        df.to_csv(output_path,index=False)
        
# extract embedding layers into csv file for exploration 
def extract_embedding_features(model,output_path):
    embdding_vec_df=None
    for layer in range(20,40):
        emb_vec=model.layers[layer].get_weights()[0]
        df =pd.DataFrame(emb_vec)
        if embdding_vec_df is None:
            embdding_vec_df=pd.DataFrame(df)
        else:
            embdding_vec_df=pd.concat([embdding_vec_df,df],axis=1)
    embdding_vec_df.to_csv(output_path,index=False)

# create instance of the model
def create_model():
    # categorical features input
    day_inp = Input(shape=(1,),dtype='int64')
    month_inp = Input(shape=(1,),dtype='int64')
    year_inp = Input(shape=(1,),dtype='int64')
    weekday_inp = Input(shape=(1,),dtype='int64')

    store_inp = Input(shape=(1,), dtype='int16')
    item_inp = Input(shape=(1,),dtype='int64')
    onpromotion_inp = Input(shape=(1,),dtype='bool')
    before_holiday_week_inp = Input(shape=(1,), dtype='bool')
    is_after_holiday_week_inp = Input(shape=(1,), dtype='bool')
    is_holiday_day_inp = Input(shape=(1,), dtype='bool')
    holiday_type_inp = Input(shape=(1,), dtype='int16')
    holiday_locale_inp = Input(shape=(1,), dtype='int16')
    holiday_locale_name_inp = Input(shape=(1,), dtype='int16')
    payment_week_inp = Input(shape=(1,), dtype='bool')

    item_family_inp = Input(shape=(1,),dtype='int16')
    perishable_inp = Input(shape=(1,),dtype='bool')
    store_city_inp = Input(shape=(1,),dtype='int16')
    store_state_inp = Input(shape=(1,), dtype='int16')
    store_type_inp = Input(shape=(1,),dtype='int16')
    store_cluster_inp = Input(shape=(1,),dtype='int16')

    # numeric data features
    numeric_0 = Input(shape=(1,),dtype='float32')
    numeric_1 = Input(shape=(1,),dtype='float32')
    numeric_2 = Input(shape=(1,),dtype='float32')
    numeric_3 = Input(shape=(1,),dtype='float32')
    numeric_4 = Input(shape=(1,),dtype='float32')
    numeric_5 = Input(shape=(1,),dtype='float32')
    numeric_6 = Input(shape=(1,),dtype='float32')
    numeric_7 = Input(shape=(1,),dtype='float32')
    numeric_8 = Input(shape=(1,),dtype='float32')
    numeric_9 = Input(shape=(1,),dtype='float32')
    numeric_10 = Input(shape=(1,),dtype='float32')
    numeric_11 = Input(shape=(1,),dtype='float32')

    # embedding layers
    day_emb = Embedding(31, 7, input_length=1, embeddings_regularizer=l2(1e-5), name='day_emb')(day_inp)
    month_emb = Embedding(12, 5, input_length=1, embeddings_regularizer=l2(1e-5), name='month_emb')(month_inp)
    year_emb = Embedding(3, 3, input_length=1, embeddings_regularizer=l2(1e-5), name='year_emb')(year_inp)
    weekday_emb = Embedding(7, 7, input_length=1, embeddings_regularizer=l1(1e-5), name='weekday_emb')(weekday_inp)

    store_emb = Embedding(54,6,input_length=1, embeddings_regularizer=l2(1e-5),name='store_emb')(store_inp)
    item_emb = Embedding(4100,16,input_length=1, embeddings_regularizer=l2(1e-5),name='item_emb')(item_inp)
    item_onpromotion_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-5),name='item_onpromotion_emb')(onpromotion_inp)

    before_holiday_week_emb = Embedding(2, 1, input_length=1, embeddings_regularizer=l2(1e-5),
                                        name='before_holiday_week_emb')(before_holiday_week_inp)
    is_after_holiday_week_emb = Embedding(2, 1, input_length=1, embeddings_regularizer=l2(1e-5),
                                          name='after_holiday_week_emb')(is_after_holiday_week_inp)
    is_holiday_day_emb = Embedding(2, 1, input_length=1, embeddings_regularizer=l2(1e-5), name='holiday_day_emb')(
        is_holiday_day_inp)
    holiday_type_emb = Embedding(7, 5, input_length=1, embeddings_regularizer=l2(1e-5), name='holiday_type_emb')(
        holiday_type_inp)
    holiday_locale_emb = Embedding(4, 4, input_length=1, embeddings_regularizer=l2(1e-5), name='holiday_locale_emb')(
        holiday_locale_inp)
    holiday_locale_name_emb = Embedding(25, 5, input_length=1, embeddings_regularizer=l1(1e-5),
                                        name='holiday_locale_name_emb')(holiday_locale_name_inp)

    payment_week_emb = Embedding(2,1,input_length=1, embeddings_regularizer=l2(1e-5),name='payment_emb')(payment_week_inp)
    item_family_emb = Embedding(33, 5, input_length=1, embeddings_regularizer=l2(1e-5), name='item_family_emb')(item_family_inp)
    item_perishable_emb = Embedding(2, 1, input_length=1, embeddings_regularizer=l2(1e-5), name='item_perishable_emb')(perishable_inp)

    store_city_emb = Embedding(22,5,input_length=1, embeddings_regularizer=l2(1e-5),name='store_city_emb')(store_city_inp)
    store_state_emb = Embedding(16,5,input_length=1, embeddings_regularizer=l2(1e-5),name='store_state_emb')(store_state_inp)
    store_type_emb = Embedding(5,5,input_length=1, embeddings_regularizer=l2(1e-5),name='store_type_emb')(store_type_inp)
    store_cluster_emb = Embedding(18,5,input_length=1, embeddings_regularizer=l2(1e-5),name='store_cluster_emb')(store_cluster_inp)

    numeric = concatenate([numeric_0,numeric_1,numeric_2,numeric_3,numeric_4,numeric_5,
                           numeric_6, numeric_7, numeric_8, numeric_9, numeric_10, numeric_11])

    x= concatenate([day_emb,month_emb,year_emb,weekday_emb,store_emb,item_emb,item_onpromotion_emb,
                     before_holiday_week_emb, is_after_holiday_week_emb,is_holiday_day_emb, holiday_type_emb,
                     holiday_locale_emb, holiday_locale_name_emb, payment_week_emb,item_family_emb,
                     item_perishable_emb,store_city_emb,store_state_emb,store_type_emb,store_cluster_emb])
    x = Flatten()(x)
    x = concatenate([x,numeric])
    x = BatchNormalization()(x)
    x = Dense(32,activation='relu',kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.5)(x)
    x = Dense(16,activation='relu',kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(16,activation='relu',kernel_regularizer=l2(1e-5))(x)
    x = Dropout(0.5)(x)
    x = Dense(1,activation='relu',kernel_regularizer=l2(1e-5))(x)
    
    inputs=[day_inp,month_inp,year_inp,weekday_inp,store_inp,item_inp,onpromotion_inp,
                      before_holiday_week_inp, is_after_holiday_week_inp,
                      is_holiday_day_inp, holiday_type_inp, holiday_locale_inp, holiday_locale_name_inp,
                      payment_week_inp,item_family_inp,perishable_inp,store_city_inp,store_state_inp,
                      store_type_inp,store_cluster_inp,numeric_0,numeric_1,numeric_2,numeric_3,numeric_4,numeric_5,numeric_6,
                      numeric_7,numeric_8,numeric_9,numeric_10,numeric_11]
    nn_model = Model(inputs,x)

    nn_model.compile(loss = mean_squared_error,optimizer='rmsprop',
                 metrics=[RootMeanSquaredError()])
    nn_model.summary()
    return nn_model

# create cat boost model
def create_catBoost_model(data_path,test_path,test_output):
    model = CatBoostRegressor(iterations=2000,
                              learning_rate=1e-2,random_state=42,
                              task_type='CPU',
                              l2_leaf_reg=1e-4, bootstrap_type='MVS',subsample=0.35)
    flag=True
    # we work on batch of 1 million rows
    for chunk in pd.read_csv(data_path, chunksize=1000000):
        target=chunk.loc[:,['unit_sales']]
        chunk.drop(columns=['unit_sales'],inplace=True)
        chunk['day']=chunk['day'].astype(int)
        chunk['weekday'] = chunk['weekday'].astype(int)
        chunk['store_nbr'] = chunk['store_nbr'].astype(int)
        chunk['item_nbr'] = chunk['item_nbr'].astype(int)
        chunk['onpromotion'] = chunk['onpromotion'].astype(int)
        chunk['type'] = chunk['type'].astype(int)
        chunk['cluster'] = chunk['cluster'].astype(int)
        X_train, X_test, y_train, y_test = train_test_split(chunk, target, test_size=0.25, random_state=42,shuffle=True)
        y_train_normalize = np.log1p(y_train)
        y_test_normalize = np.log1p(y_test)
        if flag:
            model.fit(X_train,y_train_normalize,eval_set=(X_test,y_test_normalize),
                    use_best_model=True,cat_features=['day','weekday','store_nbr','item_nbr','onpromotion','cluster','type'])
            flag=False
        else:
            # load saved model and keep training from last stop
            model.fit(X_train,y_train_normalize,
                    init_model='model.cbm',
                    eval_set=(X_test,y_test_normalize),
                    use_best_model=True,
                    cat_features=['day','weekday','store_nbr','item_nbr','onpromotion','cluster','type'])
        # save the model to cbm file
        model.save_model('model.cbm')
        
    # predict test values with the catboost model
    for chunk in pd.read_csv(test_path,chunksize=1000000):
        ids = chunk.loc[:, ['id']].copy()
        chunk.drop(columns=['id'], inplace=True)
        chunk['day'] = chunk['day'].astype(int)
        chunk['weekday'] = chunk['weekday'].astype(int)
        chunk['store_nbr'] = chunk['store_nbr'].astype(int)
        chunk['item_nbr'] = chunk['item_nbr'].astype(int)
        chunk['onpromotion'] = chunk['onpromotion'].astype(int)
        chunk['type'] = chunk['type'].astype(int)
        chunk['cluster'] = chunk['cluster'].astype(int)
        preds_norm = model.predict(chunk)
        preds = np.expm1(preds_norm)
        ids['unit_sales'] = preds
        write_chunk_to_csv(ids, test_output)

path_data = r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\catboost_train.csv"
test_out = r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\clf_predictions.csv"
test_path = r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\test_kaggle_emb.csv"
create_catBoost_model(path_data,test_path,test_out)


# file path input and output
path_to_train=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\aug_emb_train.csv"
path_to_val=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\aug_emb_test.csv"
path_to_test=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\test_kaggle_emb.csv"
output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\\nn_model_test_predictions_Aug.csv"
embdding_output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\emb_vec_Aug.csv"
preds_output=r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\nn_model_test_predictions_Aug.csv"

# create model , we difine number of rows in train and test for steps per epoch
model =create_model()
number_of_rows_train=7305460
number_of_rows_test=3131487
batch_size_train= 1024
batch_size_test= 512
# train the modek in batch, useing data generator
history = model.fit(generate_data_from_file(path_to_train,batch_size_train),
                    callbacks=set_callbacks(),
                    shuffle=True,
                    steps_per_epoch=number_of_rows_train // batch_size_train,
                    validation_data= generate_data_from_file(path_to_val,batch_size_test),
                    validation_steps=number_of_rows_test // batch_size_test,
                    epochs=25)

# plot model learning metrics for train,test
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model RMSE loss per epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# save embedding layers to csv
extract_embedding_features(model,embdding_output)

# predict test with dl model
for chunk in pd.read_csv(path_to_test,chunksize=1000000):
    ids=chunk.loc[:,['id']].copy()
    chunk.drop(columns=['id'],inplace=True)
    features=chunk.columns
    preds_norm= model.predict([chunk[f] for f in features])
    preds = np.expm1(preds_norm)
    ids['unit_sales']=preds
    write_chunk_to_csv(ids,preds_output)

# create catboost model
path_data = r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\catboost_train.csv"
test_out = r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\clf_predictions.csv"
test_path = r"C:\Users\gal\Desktop\ISE\semesterG\deep learning workshop\assignment2\Embedding\gal\test_kaggle_emb.csv"
create_catBoost_model(path_data,test_path,test_out)


