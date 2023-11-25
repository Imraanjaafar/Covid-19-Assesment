#%%
#1. Setup - mainly importing packages
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from time_series_helper import WindowGenerator

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
# %%
#2. Download and extract dataset
csv_path = r"C:\Users\Acer Nitro5\Desktop\Capstone_2\cases_malaysia_covid.csv"
# %%
# %%
#3. Load dataset with pandas
df = pd.read_csv(csv_path)
selected_columns = ['date', 'cases_new', 'cases_import', 'cases_recovered', 'cases_active']
df = df[selected_columns]

df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
#%%
print(df.info())
#%%
# Convert dtypes new_cases from obj to int
# Assuming 'cases_new' contains strings representing numbers with possible commas
df['cases_new'] = df['cases_new'].replace({',': ''}, regex=True)
 
df['cases_new'] = pd.to_numeric(df['cases_new'].str.replace(',', ''), errors='coerce', downcast='integer')
 
# Print the data types to verify the changes
print(df.dtypes)
 
# %%
df.describe().T

# %%
#4.Basic Data Inspection
plot_cols = ['cases_new', 'cases_import', 'cases_recovered', 'cases_active']
plot_features = df[plot_cols]
plot_features.index = df['date']
_ = plot_features.plot(subplots=True)

plot_features = df[plot_cols][:480]
plot_features.index =df['date'][:480]
_ = plot_features.plot(subplots=True)

#%%
#5. Data Cleaning
print(df.isnull().sum())

#%%
# Null value use median to filled
df['cases_new'].fillna(df['cases_new'].mean(), inplace=True)
print(df.isnull().sum()) # double check, is it have any null value
 
#%%
print(df.duplicated().sum()) # total duplicated = 10
print(df.shape)
#%%
# Duplicate row
df.drop_duplicates(inplace=True)

# Double check duplicate
print(df.duplicated().sum())
print(df.shape)



#%%
#6. Feature engineering
# Function to create lag features for a target variable
def create_lag_features(data, target_column, lag_steps):
    for i in range(1, lag_steps + 1):
        data[f'{target_column}_lag_{i}'] = data[target_column].shift(i)
    return data

# Example usage
target_variable = 'cases_new'
lag_steps = 7  # You can adjust this based on your needs

# Create lag features for the target variable
df = create_lag_features(df, target_variable, lag_steps)

# Drop rows with NaN values resulting from the lag operation
df = df.dropna()

# Print or inspect the updated DataFrame
print(df.head())

#%%
#8. Train validation test split
#Note : If you are doing splitting before data windowing,  make sure to not use method such as sklearn.medel.selection.train_test_split that will randomize the order of the data, because that will disrupt the time steps of the data.
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]
# %%
#9. Data normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#%%
#10. Data Inspection after normalization
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)

#%%
# Create a TensorBoard callback for single-step
log_dir_single = "logs_single/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_single = tf.keras.callbacks.TensorBoard(log_dir=log_dir_single, histogram_freq=1)
 
# Create a TensorBoard callback for multi-step
log_dir_multi = "logs_multi/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_multi = tf.keras.callbacks.TensorBoard(log_dir=log_dir_multi, histogram_freq=1)

#%%
#Scenario 1 - Single step model
single_wide_window = WindowGenerator(input_width=30, label_width=30, shift=1, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])

# %%
#12. Model development
#Create LSTM model for this scenario
lstm_model = keras.Sequential()
lstm_model.add(keras.layers.LSTM(256, return_sequences=True))
lstm_model.add(keras.layers.Dropout(0.2))
lstm_model.add(keras.layers.Dense(1))
lstm_model.add(keras.layers.Dropout(0.2))  # Add dropout layer after Dense layer
# %%
def mape(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])  # Flatten the true values
    y_pred = tf.reshape(y_pred, [-1])  # Flatten the predicted values
    return tf.reduce_mean(tf.abs((y_true - y_pred) / y_true))
 
MAX_EPOCHS = 20
 
def compile_and_fit(model, window, patience=3, callbacks_list=None):
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        mode='min'
    )
 
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(), mape]
    )
 
    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS,
        validation_data=window.val,
        callbacks=[early_stopping] + (callbacks_list or [])
    )

# %%
# Compile the model and train
history_1 = compile_and_fit(lstm_model, single_wide_window, callbacks_list=[tensorboard_callback_single])
# %%
#Evaluate the model
print(lstm_model.evaluate(single_wide_window.val))
print(lstm_model.evaluate(single_wide_window.test))
# %%
# Plot the resultt
single_wide_window.plot(model=lstm_model, plot_col='cases_new')
#%%
#Display model summary
lstm_model.summary()
#Display model structure
tf.keras.utils.plot_model(lstm_model)
# %%
#Scenario 2 - Multi step model
multi_wide_window = WindowGenerator(input_width=30, label_width=30, shift=30, train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['cases_new'])
# %%
#Build multi step model
multi_lstm = keras.Sequential()
multi_lstm.add(keras.layers.LSTM(256, return_sequences=False))
multi_lstm.add(keras.layers.Dropout(0.2))
multi_lstm.add(keras.layers.Dense(30*1))
multi_lstm.add(keras.layers.Dropout(0.2))  # Additional dropout layer
multi_lstm.add(keras.layers.Reshape([30, 1]))

# %%
#Compile and train model
history_2 = compile_and_fit(multi_lstm, multi_wide_window, callbacks_list=[tensorboard_callback_multi])
# %%
print(multi_lstm.evaluate(multi_wide_window.val))
print(multi_lstm.evaluate(multi_wide_window.test))
# %%
multi_wide_window.plot(model=lstm_model, plot_col='cases_new')

#%%
#Display model summary
multi_lstm.summary()
#Display model structure
tf.keras.utils.plot_model(multi_lstm)