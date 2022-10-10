import geemap, ee
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
try:
        ee.Initialize()
except Exception as e:
        ee.Authenticate()
        ee.Initialize()

# satelites data (average night light)
viirs = ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG").filter(ee.Filter.date('2021-01-01', '2021-12-31')).select('avg_rad')
# cite names (sort by population)
with open('cities.dat') as sample:
    select_cities=sample.read().splitlines()

print(select_cities)
f = ee.Filter

#f = f.stringContains('ADM1_NAME','Alexandria')

# administrative geo data of cities 
f = ee.Filter.inList('ADM2_NAME', select_cities)
city_geoms = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level2").filter(f)


#print(city_geoms.select('ADM1_NAME').first().getInfo());

def get_city_avg_rad(img):
    return img.reduceRegions(reducer=ee.Reducer.mean(), collection=city_geoms, scale=500)

# function to get individual img dates
def get_date(img):
    return img.set('date', img.date().format())

# map these functions to our image collection
reduced_cities = viirs.map(get_city_avg_rad).flatten()
dates = viirs.map(get_date)

# get lists
key_cols = ['ADM2_NAME','mean']
cities_list = reduced_cities.reduceColumns(ee.Reducer.toList(len(key_cols)), key_cols).values()
dates_list = dates.reduceColumns(ee.Reducer.toList(1), ['date']).values()

# some numpy maneuvers to structure our data
df = pd.DataFrame(np.asarray(cities_list.getInfo()).squeeze(), columns=key_cols)
dates = np.asarray(dates_list.getInfo()).squeeze()

df['mean'] = df['mean'].astype(float)

means = df.groupby('ADM2_NAME')['mean'].mean().sort_values(ascending=False)

# generate the file: result-202X.txt (In some of the countries, there are strange administrative data, so we check ADM1_NAME and ADM2_NAME together to generate 2 files.)
print(means)