import numpy as np
from datetime import datetime, timedelta

def is_nan(nan):
    return nan != nan

def load_paths(Cityname, cell_size, poi_mapping=''):
    for file_type in ['train','valid','test']:
        trajs = np.load(f'dataset/{Cityname}_data/traj_{file_type}.npy')
        print("Trajectories Number: {}".format(trajs.shape[0]))
        paths = []
        for i in range(trajs.shape[0]):
            tmp = trajs[i]
            ori = tmp[0]
            dest = tmp[2]
            if (is_nan(map_segment_to_poi(ori,poi_mapping))==False) and (is_nan(map_segment_to_poi(dest,poi_mapping)) == False):
                paths.append([ori,tmp[1],dest,round(tmp[3], 2) ])

        print("Paths Number: {}".format(len(paths)))
        np.save(f'dataset/{Cityname}_data/traj_{file_type}_{cell_size}.npy', paths)

def timestamp_to_HHMM(timestamp):
    return (datetime.utcfromtimestamp(timestamp) + timedelta(hours=8)).strftime('%H:%M')

# Function to extract segment ID and timestamp from the description
def extract_segment_id_and_timestamp(poi_mapping, use_multiple = False):
    def inner(values):
        return [map_segment_to_poi(values[0],poi_mapping, use_multiple),timestamp_to_HHMM(float(values[1])), 
                map_segment_to_poi(values[2],poi_mapping, use_multiple), values[3]]
    return inner

def get_bins(values, distance_bins):
    for i in range(1, len(distance_bins)): # find the associate bin
        if distance_bins[i] >= values:
            values = i - 1
            break
    return values

def get_hierachy(poi):
    if '房产小区' in poi:
        return poi.split(':')[1]
    else:
        return poi.split(':')[0]

def get_poi(segment_str, use_multiple=False):
    if is_nan(segment_str):
        return float('NaN')
    if use_multiple == False:
        segment_poi = get_hierachy(segment_str.split(',')[0])
    else: # use multiple
        segment_str = segment_str.split(',')
        if len(segment_str) > 1:
            segment_poi = [get_hierachy(single)for single in segment_str]
        else:
            segment_poi = get_hierachy(segment_str[0])

    return segment_poi

def cell_center(point, min_x, min_y ,cell_size):
    """
    Calculate the center point of the cell in which the given point resides.
    Assumes point is a shapely Point object with (longitude, latitude) coordinates.
    """
    # Approximate conversion factor from degrees to meters at Beijing's latitude
    conversion_factor = 111000

    # Adjust boundaries to be multiples of cell_size
    grid_min_x = min_x - (min_x % cell_size)
    grid_min_y = min_y - (min_y % cell_size)

    x_index = int((point.y* conversion_factor - min_x) // cell_size)
    y_index = int((point.x* conversion_factor - min_y) // cell_size)
    
    cell_x = ((point.y *conversion_factor  - grid_min_x) // cell_size) * cell_size + cell_size / 2 + grid_min_x
    cell_y = ((point.x *conversion_factor  - grid_min_y) // cell_size) * cell_size + cell_size / 2 + grid_min_y
    
    # Convert back to degrees, index
    return (cell_x / conversion_factor, cell_y / conversion_factor), (x_index, y_index)

# Function to map segment ID to POI type
def map_segment_to_poi(segment_id, poi_mapping, use_multiple= False):
    segment_id = int(segment_id)
    segment_str = poi_mapping.get(segment_id, 'Unknown')
    return get_poi(segment_str, use_multiple)

if __name__ == '__main__':
    print(timestamp_to_HHMM(1475471927.198))