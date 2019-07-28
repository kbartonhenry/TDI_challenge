import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#import & format
df = pd.read_csv('Downloads/POWER_SinglePoint_Daily_20150101_20150305_041d39N_002d15E_4900a583.csv')
df['date'] = df.apply(lambda x:dt.strptime("{0} {1} {2} 00:00:00".format(int(x['YEAR']),int(x['MO']),int(x['DY'])), "%Y %m %d %H:%M:%S"),axis=1)

#solar irradiance

#plot 
plt.title('Downward Thermal Infrared (Longwave) Radiative Flux')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.plot(df['date'],df['ALLSKY_SFC_LW_DWN'])
plt.gcf().autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('kW-hr/m^2/day')
plt.save('radiative_flux.png')

#plot 
plt.title('Isolation Clearness Index')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.plot(df['date'],df['KT'])
plt.gcf().autofmt_xdate()
plt.xlabel('Date')
plt.ylabel('(dimensionless)')
plt.save('clearness.png')