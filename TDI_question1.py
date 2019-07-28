import pandas as pd 
import numpy as np
from difflib import SequenceMatcher
import itertools
import string

df = pd.read_csv('Parking_Citations.csv')

#Quick data exploration
df.info()
df.describe()
print ((df.isnull().sum() / len(df)) * 100)

#I am interpreting '10 digits of precision' as 10 decimil places (not total significant digits). 10 significant digits did not make empirical sense for some questions (Q3)

#Q1: Mean violantion fine for all citations
print('For all citations, the mean violation fine is %.10f USD' %df.ViolFine.mean())

#Q2: district with highest mean violantion fine
print (df.PoliceDistrict.unique())
df.PoliceDistrict = df.PoliceDistrict.str.lower()
df.PoliceDistrict.replace({'notheastern':'northeastern'}, inplace = True)
dist_rank = df.groupby('PoliceDistrict')['ViolFine'].mean().sort_values(ascending =False).reset_index()
print('The district with the highest mean violation fine is %s  with a mean fine of %.10f USD' %(dist_rank.PoliceDistrict[0], dist_rank.ViolFine[0]))

#Q3: total citations given each year (2004-2014). Linear Regression (total citations, year); slope.
from datetime import datetime as dt
df.ViolDate = df.ViolDate.fillna('01/01/9999 00:00:00 AM')
df['viol_year'] = df.ViolDate.apply(lambda x: dt.strptime(x, '%m/%d/%Y %H:%M:%S %p').year)
time_df = df[(df.viol_year >= 2004) & (df.viol_year <= 2014)]
print ('Total number of citations per year:')
print(time_df.groupby('viol_year')['Citation'].sum())

import statsmodels.api as sm
grp_df = time_df.groupby('viol_year')['Citation'].sum().reset_index()
model = sm.OLS(grp_df.Citation,sm.add_constant(grp_df.viol_year))
results = model.fit()
print(results.summary())
print ('The coeffient on year (slope) is %.10f' %(results.params[1]))


#Q4: only open penalty fees: 81st precentile $
open_df = df[(df.OpenPenalty > 0)]
print ('The value at the 81st percentile is: %.10f' %np.percentile(open_df.OpenPenalty, 81))

#Q5: 
'''
def match_words(a,b):
	if SequenceMatcher(None, a, b).ratio() > .6:
		if len(a) > len(b):
			return a
		else: return b
	else: pass

df_2017 = df[(df.viol_year == 2017)]

#get rid of Make Nulls
print ((df_2017.isnull().sum() / len(df_2017)) * 100)
df_2017.Make = df_2017.Make.fillna('')
df_2017.Make = df_2017.Make.apply(lambda x: x.strip())

makes = sorted(list(df_2017.Make.unique()))

letter_groups = {}
for letter in list(string.ascii_lowercase):
	letter_groups[letter] = list(filter(lambda x: x.lower().startswith(letter), makes))
maker_list = list()
for key in letter_groups.keys():
	for a, b in itertools.combinations(letter_groups[key], 2):
		if match_words(a, b):
			maker_list.append(match_words(a, b))

clean_list = []
for x, y in itertools.combinations(set(maker_list),2):
	if match_words(x,y):
		clean_list.append(match_words(x,y))

final_list = []
for x, y in itertools.combinations(set(clean_list),2):
	if match_words(x,y):
		final_list.append(match_words(x,y))

set(final_list)

'''


#Q6: 

df_theft = pd.read_csv('BPD_Part_1_Victim_Based_Crime_Data.csv')
df_theft['year'] = df_theft.CrimeDate.apply(lambda x: dt.strptime(x, '%m/%d/%Y').year)

#instances of auto theft
dft_2015 = df_theft[(df_theft.year == 2015) & (df_theft.Description == 'AUTO THEFT') ]
print ('Total number of auto thefts in 2015, by district: ')
print (dft_2015.groupby('District')['Total Incidents'].sum())


#parking citations
print ('Total number of parking citations in 2015, by district: ')
print (df.groupby('PoliceDistrict')['Citation'].count())

#ratio of auto thefts to parking citaitons for each. highest?
auto_theft = dft_2015.groupby('District')['Total Incidents'].sum().reset_index()
auto_theft.District = auto_theft.District.str.lower()
park_cite = df.groupby('PoliceDistrict')['Citation'].count().reset_index()
auto_theft.rename({'Total Incidents':'ratio'},axis = 'columns', inplace =True)
park_cite.rename({'Citation':'ratio'},axis = 'columns', inplace =True)
print ('The highest ratio of auto thefts to parking citations was in the %s district with a ratio of %.10f thefts to citations' %(auto_theft.set_index('District').div(park_cite.set_index('PoliceDistrict').sum()).sort_values(by='ratio', ascending=False).reset_index().District[0], auto_theft.set_index('District').div(park_cite.set_index('PoliceDistrict').sum()).sort_values(by='ratio', ascending=False).reset_index().ratio[0]))


