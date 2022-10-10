# memtic_location

## prepare_data

- run.py

Get global satelates night lights data from VIIRS (google earth api);
Get world administrative areas data from Global Administrative Unit Layers (GAUL);
Get top 100 largest cities by population from https://www.macrotrends.net/cities/largest-cities-by-population;

## process csv file to Memitic

- process.py 

Merge them to generate:

1. final_data.csv  -> cities with their own lights, population, etc..
2. distance_data.csv  -> distance data for each others.

## Memitic

- memtic_single.py

non-parallel version Memtic.