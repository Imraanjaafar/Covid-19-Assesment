# Covid-19-Assesment

##  Deep learning model using LSTM neural network to predict new cases (cases_new) in Malaysia using the past 30 days of number of cases.

The year 2020 marked a global catastrophe with the emergence of COVID-19, a pandemic that impacted over 200 countries. Governments implemented various measures such as travel restrictions, quarantines, and lockdowns to curb the virus's spread. However, these efforts were hindered by a lackadaisical attitude, contributing to widespread infection and loss of lives. Scientists attributed the pandemic's severity to the absence of AI-assisted tracking systems. In response, they advocated for the use of deep learning models, particularly LSTM neural networks, to predict daily COVID-19 cases. Your assignment involves creating such a model to forecast new cases in Malaysia based on the past 30 days' data. The dataset used in this project are from https://github.com/MoH-Malaysia/covid19-public.

## Directory Structure
- [Imran_Capstone_2.py](https://github.com/Imraanjaafar/Covid-19-Assesment/blob/main/Imran_Capstone_2.py)

This code appears to be related to time series analysis, specifically using TensorFlow and Keras to build and train LSTM (Long Short-Term Memory) models for forecasting COVID-19 cases in Malaysia. Here's a generalized explanation of the code:

### Import Libraries:

- The code starts by importing various libraries such as TensorFlow, Pandas, Matplotlib, and Seaborn for data manipulation, visualization, and machine learning.

### Download and Load Dataset:

- The code specifies the path to a CSV file containing COVID-19 cases data for Malaysia and loads it into a Pandas DataFrame.
- Selected columns related to date and different case types are extracted.

### Data Preprocessing:

- The 'date' column is converted to a datetime format.
- The 'cases_new' column, representing new cases, is cleaned by removing commas and converting it to numeric values.
- Basic data inspection is performed, and visualizations of different case types are plotted over time.

### Data Cleaning:

- Null values in the 'cases_new' column are filled with the mean value.
- Duplicate rows in the dataset are removed.

### Feature Engineering:

- Lag features are created for the target variable 'cases_new,' which means creating columns with the past values of 'cases_new' to be used as features in the model.

### Train-Validation-Test Split:

- The dataset is split into training, validation, and test sets, ensuring that the temporal order of the data is maintained.

### Data Normalization:

- The dataset is normalized using mean and standard deviation values calculated from the training set.

### Data Inspection after Normalization:

- Violin plots are used to visualize the distribution of normalized data.

### TensorBoard Callbacks:

- TensorBoard callbacks are set up for logging training progress.

### Single-Step Model Development:

- An LSTM model is created for the scenario of predicting the next day's cases.
- The model is compiled, trained, and evaluated on the validation and test sets.
- Evaluation metrics such as Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE) are used.

### Plotting Results:

- The model's predictions are plotted against the actual values.

### Multi-Step Model Development:

- Another LSTM model is created for the scenario of predicting multiple days ahead.
- The model is compiled, trained, and evaluated similarly to the single-step model.

### Plotting Multi-Step Results:

- The multi-step model's predictions are plotted against the actual values.

### Display Model Summary and Structure:

- The summary and structure of both the single-step and multi-step models are displayed.

## Results
- The Single-Step Model achieved a Mean Absolute Percentage Error (MAPE) error of 0.51, while the Multi-Step Model exhibited a MAPE error of 0.94. These results underscore the model's effectiveness in forecasting daily COVID cases.

## Model Architecture

- This section provides an overview of the architectures of two models: the Single-Step Model and the Multi-Step Model.

### Single Step Model
![Architecture_SingleStep](https://github.com/Imraanjaafar/Covid-19-Assesment/assets/151133555/59fddb99-4b47-4d62-ac5b-b4829ad59af8)

### Multi Step Model
![Architecture_MultipleStep](https://github.com/Imraanjaafar/Covid-19-Assesment/assets/151133555/9601a95f-40ad-4796-8064-e12809908eac)


