import pandas as pd  # https://www.youtube.com/watch?v=mgX0Iz4q0bE
from neuralprophet import NeuralProphet
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv('weatherAUS.csv')
melb = df[df['Location'] == 'Melbourne']
melb['Date'] = pd.to_datetime(melb['Date'])

#  taking the data before the gap
melb['Year'] = melb['Date'].apply(lambda x: x.year)
melb = melb[melb['Year'] <= 2015]
# plt.plot(melb['Date'], melb['Temp3pm'])


data = melb[['Date', 'Temp3pm']]  # 2nd variable is the one we want to predict
data.dropna(inplace=True)
data.columns = ['ds', 'y']

m = NeuralProphet()

model = m.fit(data, freq='D', epochs=100)
future = m.make_future_dataframe(data, periods=10000)
forecast = m.predict(future)
# with open('saved_model.pkl', "wb") as f:
#     pickle.dump(m, f)
# to save the plot

plot1 = m.plot(forecast)
plot2 = m.plot_components(forecast)
plt.show()
