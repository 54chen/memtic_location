# mimetic_location

## prepare_data

- run.py

Get global satelates night lights data from VIIRS (google earth api);
Get world administrative areas data from Global Administrative Unit Layers (GAUL);
Get top 100 largest cities by population from https://www.macrotrends.net/cities/largest-cities-by-population;

## process csv file to Mimetic

- process.py 

Merge them to generate:

1. final_data.csv  -> cities with their own lights, population, etc..
2. distance_data.csv  -> distance data for each others.

## Mimetic

- mimetic_single.py

Single-threaded version Mimetic

- mimetic_async.py

Separated version Mimetic

- mimetic_async_separate.py

Parallel evaluation version Mimetic

## Visualisation
- only support jupyter notebook

- how to start visualisation?

```shell
#jupyter notebook
```

Then run on your  brower.

 ![alt text](https://raw.githubusercontent.com/54chen/mimetic_location/main/result1.png)