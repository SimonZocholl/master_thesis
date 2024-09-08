# %%
import sys

# %%
from srai.regionalizers.geocode import geocode_to_region_gdf
from sqlalchemy import create_engine, text

import geopandas as gpd
from tqdm import  tqdm
import pandas as pd
import os
import csv



# %%
pgsql_user = ""
pgsql_passwort = ""
pgsql_server = ""

pgsql_port = "5432"
pgsql_db = "mobtrack"

# %%
database_file = 'postgresql://' + pgsql_user + ':' \
                + pgsql_passwort + '@' \
                + pgsql_server + ':' \
                + pgsql_port + '/' \
                + pgsql_db

engine = create_engine(database_file, echo=False)

# %%
areaName = "Munich"
area = geocode_to_region_gdf(areaName)
    
select_txt = "sl.id, sl.user_id, sl.started_at, sl.finished_at, sl.length, sl.mode, sl.geom, sl.confirmed, sd.gender, sd.age, sd.income_net, sd.job ,sd.education, sd.hh_size , sd.hh_children, sd.usage_car, sd.activity_work, sd.activity_leisure, sd.car, sd.activity_errand, sd.distance_work , sd.distance_leisure , sd.distance_errand"
from_txt = "mobilitaet_leben.storyline sl join mobilitaet_leben.survey_data sd on sl.user_id  = sd.user_id "
where_txt = "ST_GeometryType(sl.geom) != 'ST_Point' " \
                +"AND (sl.mode=10 or sl.mode = 2 or sl.mode = 8 or sl.mode = 10 or sl.mode = 11 or sl.mode = 18)" \
                +"AND ST_Within(sl.geom, ST_GeomFromText('SRID=4326;"+area.geometry[0].wkt+"'))"
query = f"SELECT {select_txt} FROM {from_txt} WHERE {where_txt}"
query

# %%
def get_data_from_db(engine, update=True, areaName=None):
    
    if areaName is None:
        areaName=config.area_name

    path = os.path.join(os.path.curdir, "data")
    name = "mydata_raw.csv"
    area = geocode_to_region_gdf(areaName)
    
    select_txt = "sl.id, sl.user_id, sl.started_at, sl.finished_at, sl.length, sl.mode, sl.geom, sl.confirmed, sd.gender, sd.age, sd.income_net, sd.job ,sd.education, sd.hh_size , sd.hh_children, sd.usage_car, sd.activity_work, sd.activity_leisure, sd.car, sd.activity_errand, sd.distance_work , sd.distance_leisure , sd.distance_errand"
    from_txt = "mobilitaet_leben.storyline sl join mobilitaet_leben.survey_data sd on sl.user_id  = sd.user_id "
    where_txt = "ST_GeometryType(sl.geom) != 'ST_Point' " \
                +"AND (sl.mode=10 or sl.mode = 2 or sl.mode = 8 or sl.mode = 10 or sl.mode = 11 or sl.mode = 18)" \
                +"AND ST_Within(sl.geom, ST_GeomFromText('SRID=4326;"+area.geometry[0].wkt+"'))"
    
    query = text(f"SELECT {select_txt} FROM {from_txt} WHERE {where_txt}")
    
    with engine.connect() as con:    
        gdf = gpd.read_postgis(query, con, geom_col="geom")  
        os.makedirs(path, exist_ok=True)
        gdf.to_csv(os.path.join(path, name))

get_data_from_db(engine)


