import ray
import pandas as pd
import numpy as np
import aiohttp
import asyncio
from tqdm.asyncio import tqdm  # 注意使用异步版本的tqdm
from .utils import load_paths, extract_segment_id_and_timestamp, cell_center
import json
import geopandas as gpd

key = ''

# 定义一个异步函数来处理 HTTP 请求
async def fetch_poi_async(session, url):
    async with session.get(url) as response:
        r = await response.text()
        return json.loads(r)['result']['pois']

@ray.remote
def getPOIBypoint(point):
    x, y = point
    async def get_poi():
        async with aiohttp.ClientSession() as session:
            url = f'https://apis.map.qq.com/ws/geocoder/v1/?location={x},{y}&key={key}&get_poi=1&poi_options=radius=500;policy=5'
            query = await fetch_poi_async(session, url)
            if query[0]['_distance'] <= 30:
                if not query[0]['category'].startswith("地名地址"):  # 地名地址:行政地名
                    return query[0]['category']
                else:
                    return 'None'
            else:
                return ",".join(single['category'] for single in query)

    # 在同步函数中运行异步代码
    try:
        result = asyncio.run(get_poi())
    except Exception as e:
        result = 'None'
    return result

def generateEdgesByCity(Cityname,cell_size):
    edges = gpd.read_file(f'dataset/{Cityname}_data/map/edges.shp')

    ## Get min_x, min_y
    points_meters = [(point.y * 111000, point.x * 111000) for point in edges['geometry'].centroid]
    min_x = min(points_meters, key=lambda x: x[0])[0]
    min_y = min(points_meters, key=lambda x: x[1])[1]
    
    POI = []
    poi_cache = {}  # Cache for storing POI data by cell center
    
    with tqdm(total=len(edges)) as pbar:
        futures = []
        for point in edges['geometry'].centroid:
            cell_center_point, cell_index = cell_center(point, min_x, min_y, cell_size)
            if cell_index in poi_cache:
                # Use the cached POI data
                poi = poi_cache[cell_index]
                futures.append((poi, pbar, cell_index))  # Mark as cached
            else:
                # Submit new task to get POI data
                future = getPOIBypoint.remote(cell_center_point)
                futures.append((future, pbar, cell_index))  # Mark as not cached
        
        for future, bar, cell_index in futures:
            bar.update()
            
            if cell_index in poi_cache:
                poi = ray.get(future)
            else:
                poi = ray.get(future)
                poi_cache[cell_index] = poi
            POI.append(poi)
    
    edges['POI_type'] = POI
    edges.drop('geometry', axis=1).to_csv(f'dataset/{Cityname}_data/edges_with_POI.csv', index=False)

    # Convert the dictionary to a list of tuples, each representing a row
    data_list = [(index, poi) for index, poi in poi_cache.items()]
    # Create a DataFrame
    df = pd.DataFrame(data_list, columns=["index", "POI"])

    # Save the DataFrame to a CSV file
    df.to_csv(f'dataset/{Cityname}_data/cells_with_poi.csv', index=False)

def gentrajByCity(Cityname,cell_size):

    poi_df = pd.read_csv('dataset/' + Cityname + f'_data/edges_with_POI.csv')
    poi_mapping = poi_df['POI_type'].to_dict()
    load_paths(Cityname, cell_size, poi_mapping=poi_mapping)

def prepare_instruction(args, type):
    traj_np = np.load('dataset/' + args.city + f'_data/traj_{type}_{args.cell_size}.npy')
    
    # Load the POI type CSV file
    poi_df = pd.read_csv('dataset/' + args.city + f'_data/edges_with_POI.csv')
    poi_mapping = poi_df['POI_type'].to_dict()
    print("Traj Num:" + str(len(traj_np)))
    traj_np = map(extract_segment_id_and_timestamp(poi_mapping, use_multiple = args.use_multiple),traj_np)
    traj_df = pd.DataFrame(traj_np, columns=['ori_id', 'ori_time', 'dest_id','dest_dist'])
    return traj_df

if __name__ == '__main__':
    for Cityname in ['bj','xa','cd']:
        gentrajByCity(Cityname, cell_size = 2000)
        # generateEdgesByCity(Cityname)
    # for Cityname in ['bj','xa','cd']:
    #     edges = gpd.read_file(f'dataset/{Cityname}_data/map/edges.shp')

    #     ## Get min_x, min_y
    #     points_meters = [(point.y, point.x) for point in edges['geometry'].centroid]
    #     min_x = min(points_meters, key=lambda x: x[0])[0]
    #     min_y = min(points_meters, key=lambda x: x[1])[1]
    #     max_x = max(points_meters, key=lambda x: x[0])[0]
    #     max_y = max(points_meters, key=lambda x: x[1])[1]
    #     print(min_x, max_x, min_y, max_y)
    ## BEIJING: 2035m*2465m
    ## XIAN:571m* 500m
    ## CD: 500m* 743m