#Modülleri ekleyelim.
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from datetime import datetime, timedelta
import pandas as pd 
from deep_model import DeepModelTS
import yaml
import os

with open(f'{os.getcwd()}\\conf.yml') as file:
    conf = yaml.load(file, Loader=yaml.FullLoader)

#Verileri okuyalım.
d = pd.read_csv('yillar.csv')
d.head()
print(d.head(5))
d['Datetime'] = [datetime.strptime(x, '%Y-%m') for x in d['Datetime']]

#Yinelenen veri varmı kontrol ediyoruz Eğer varsa ortalamasını alacağız.
d = d.groupby('Datetime', as_index=False)['veri'].mean()

# Verileri sıralama
d.sort_values('Datetime', inplace=True)

# Derin öğrenme sınıfının başlatılması
deep_learner = DeepModelTS(
    data=d, 
    Y_var='veri',
    lag=conf.get('lag'),
    LSTM_layer_depth=conf.get('LSTM_layer_depth'),
    epochs=conf.get('epochs'),
    train_test_split=conf.get('train_test_split') # Doğrulamak için kullanacağımız veriler
)

model = deep_learner.LSTModel()
yhat = deep_learner.predict()

if len(yhat) > 0:

  #Veri çerçevesini oluşturma
    fc = d.tail(len(yhat)).copy()
    fc.reset_index(inplace=True)
    fc['forecast'] = yhat

    
#Tahminlerin grafiğini çizelim.
    plt.figure(figsize=(24, 16))
    for dtype in ['veri', 'forecast']:
        plt.plot(
            'Datetime',
            dtype,
            data=fc,
            label=dtype,
            alpha=0.8
        )
    print(fc)
    plt.legend()
    plt.grid()
    plt.show()   
    

#Modelin tüm verileri kullanarak oluşturulması ve ileride tahmin yapılması
deep_learner = DeepModelTS(
    data=d, 
    Y_var='veri',
    lag=1,
    LSTM_layer_depth=64,
    epochs=2000,
    train_test_split=0 
)


deep_learner.LSTModel()

n_ahead = 168
yhat = deep_learner.predict_n_ahead(n_ahead)
yhat = [y[0][0] for y in yhat]

# Tahmin için veri çerçevesini oluşturma
fc = d.tail(400).copy() 
fc['type'] = 'original'

last_date = max(fc['Datetime'])
hat_frame = pd.DataFrame({
    'Datetime': [last_date + timedelta(days=x + 1) for x in range(n_ahead)], 
    'veri': yhat,
    'type': 'forecast'
})

fc = fc.append(hat_frame)
fc.reset_index(inplace=True, drop=True)

plt.figure(figsize=(12, 15))
for col_type in ['original', 'forecast']:
    plt.plot(
        'Datetime', 
        'veri', 
        data=fc[fc['type']==col_type],
        label=col_type
        )
print(fc[fc['type']==col_type])
plt.legend()
plt.grid()
plt.show()    