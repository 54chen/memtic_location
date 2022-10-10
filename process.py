import csv
import geopy.distance
import pandas as pd
from geopy.geocoders import Nominatim
from os.path import exists

FINAL_DATA_CSV='final_data.csv'
DISTANCE_DATA_CSV='distance_data.csv'

with open("2022.csv", encoding='utf-8-sig') as file_name:
    result2022 = list(csv.reader(file_name))


with open("2022-2.csv", encoding='utf-8-sig') as file_name:
    result2022bak = list(csv.reader(file_name))

with open("2021.csv", encoding='utf-8-sig') as file_name:
    result2021 = list(csv.reader(file_name))

with open("2021-2.csv", encoding='utf-8-sig') as file_name:
    result2021bak = list(csv.reader(file_name))

with open("population.csv", encoding='utf-8-sig') as file_name:
    population = list(csv.reader(file_name))

def normalise_main_data(df):
    cols_to_norm = ['development', 'increase', 'population']
    df[cols_to_norm] = df[cols_to_norm].astype(float)
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

def normalise_distance_data(df):
    cols_to_norm = ['distance']
    df[cols_to_norm] = df[cols_to_norm].astype(float)
    df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df

def merge(result2022, result2022bak):
    new_2022 = []
    cities = []
    for city in result2022:
        for city_bak in result2022bak:
            if city[0] == city_bak[0] and float(city[1]) <= float(city_bak[1]):
                new_2022.append(city_bak)
                cities.append(city_bak[0])
            if city[0] == city_bak[0] and float(city[1]) > float(city_bak[1]):
                new_2022.append(city)
                cities.append(city[0])

    for city in result2022:
        if city[0] not in cities:
            new_2022.append(city)
            cities.append(city[0])
    for city_bak in result2022bak:
        if city_bak[0] not in cities:
            new_2022.append(city_bak)
            cities.append(city_bak[0])
    return new_2022


def get_population(ct):
    for city in population:
        if ct == city[1]:
            return city[3].replace(',','')
    print('city population not found', ct)
    return 0


def generate_final_data(new_data_2022, new_data_2021):
    final_data = []
    for city in new_data_2022:
        for city2 in new_data_2021:
            if city[0] == city2[0]:
                row = []
                row.append(city[0])  # city name
                row.append(city[1])  # recent 1 year development
                row.append((float(city[1]) - float(city2[1]))/float(city2[1])) # increase rate
                row.append(get_population(city[0]))

                geolocator = Nominatim(user_agent="MyApp")
                city_name_for_geopy = city[0]
                if city[0] == 'Tookyoo':
                    city_name_for_geopy = 'tokyo'

                location = geolocator.geocode(city_name_for_geopy)
                print(city[0])
                row.append(location.latitude)
                row.append(location.longitude)
                final_data.append(row)

    df = pd.DataFrame(final_data, columns=['name', 'development', 'increase', 'population', 'latitude', 'longitude']).sort_values(
        by='increase', ascending=False, ignore_index=True)
    df = normalise_main_data(df)
    df.to_csv(FINAL_DATA_CSV)
    return df


if not exists(FINAL_DATA_CSV):
    new_data_2022 = merge(result2022, result2022bak)
    new_data_2021 = merge(result2021, result2021bak)

    df = generate_final_data(new_data_2022, new_data_2021)
else:
    df = pd.read_csv(FINAL_DATA_CSV)

df = df.reset_index()  # make sure indexes pair with number of rows
distance = []
uniq = []
for index, row in df.iterrows():
    for index, row2 in df.iterrows():
        uniq_str = row['name']+"_"+row2['name']
        print('distance :', uniq_str)
        r = []
        
        if row['name'] < row2['name']:
            uniq_str = row2['name']+"_"+row['name']
        if uniq_str in uniq:
            continue
        
        r.append(row['name'])  # city1
        r.append(row2['name'])  # city2
        r.append(geopy.distance.geodesic(
            (row['latitude'], row['longitude']), (row2['latitude'], row2['longitude'])).km)  # distance
        distance.append(r)
        uniq.append(uniq_str)
df2 = pd.DataFrame(distance, columns=['city1', 'city2', 'distance']).sort_values(
    by='distance', ascending=False, ignore_index=True)
df2 = normalise_distance_data(df2)
df2.to_csv(DISTANCE_DATA_CSV)
