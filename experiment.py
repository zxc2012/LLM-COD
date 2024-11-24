import colorsys

import folium
import folium.plugins as plugins
import branca.colormap as cm
import pandas as pd
from shapely.geometry import Polygon
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import geopandas as gpd
from tqdm import tqdm
from preprocess import cell_center
import torch

def find_closest_vector(predicted_vector, cells, end_cell, alpha):
    # Pre-convert cells to tensors and move them to GPU if possible
    cell_tensors = {cell_coords: torch.tensor(cell_vector).to('cuda')
                    for cell_coords, cell_vector in cells.items()}

    # Use tensor broadcasting to calculate all distances at once
    distances = {cell_coords: torch.nn.CrossEntropyLoss()(predicted_vector, cell_tensor)
                for cell_coords, cell_tensor in cell_tensors.items()}
    # distances = {cell_coords: torch.nn.CrossEntropyLoss(weight=torch.tensor([1.63, 2.67, 4.93, 13.56, 1.17, 3.06, 4.61, 1.89, 4.6, 14.92, 5.53,
    #                                                               9.07, 1.0, 7.15, 3.1, 4.44, 10.77, 13.35, 2.89, 20.04, 372.99, 181.58,
    #                                                               587.47], device=predicted_vector.device))(predicted_vector, cell_tensor)
    #              for cell_coords, cell_tensor in cell_tensors.items()}

    # Find the closest cell by minimum distance
    closest_cell = min(distances, key=distances.get)
    if str(end_cell) not in distances or distances[str(end_cell)] - distances[closest_cell] < alpha:
        closest_cell = end_cell
    return closest_cell

def generate_high_contrast_colors(n):
    colors = []
    step = 360.0 / n  # Divide the color wheel into n parts
    for i in range(n):
        hue = step * i / 360.0
        lightness = 0.5  # Middle lightness to avoid too dark or too light colors
        saturation = 0.8  # High saturation for vivid colors
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#{:02X}{:02X}{:02X}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors

def plot_rmse_by_od_flow(plot_sample, names, interval = 30):
    model_names = names
    # Extract OD flow values
    od_flows = np.array(plot_sample[0])
    # Sort the samples by OD flow values
    sorted_indices = np.argsort(od_flows)
    od_flows_sorted = od_flows[sorted_indices]

    # Define the range of OD flow values for each interval
    min_flow = int(np.min(od_flows))
    max_flow = int(np.max(od_flows))
    intervals = range(min_flow, max_flow + interval, interval)

    plt.figure(figsize=(11, 8), layout='constrained')

    # Define colors for each model
    colors = generate_high_contrast_colors(len(plot_sample) - 1)
    markers = ['o', 'x', 'p', 'D', 'P']  # Define different markers for each model
    # Plotting the RMSE against maximum OD flow values
    for model_idx, model in enumerate(plot_sample[1:]):
        errors = np.array(model)
        model_names[model_idx] += ' RMSE:' + str(np.round(np.sqrt(np.mean(errors**2)), 2))
        errors_sorted = errors[sorted_indices]

        # Initialize arrays to store interval RMSE and maximum OD flow values
        interval_rmse = []
        max_od_flow_values = []

        # Calculate RMSE for each interval
        for start_flow in intervals:
            end_flow = start_flow + interval
            mask = (od_flows_sorted >= start_flow) & (od_flows_sorted < end_flow)
            if np.any(mask):
                interval_errors = errors_sorted[mask]
                rmse = np.sqrt(np.mean(interval_errors**2))
                interval_rmse.append(rmse)
                max_od_flow = np.max(od_flows_sorted[mask])
                max_od_flow_values.append(max_od_flow)
        # Plot each model's RMSE with a unique color and marker
        plt.plot(max_od_flow_values, interval_rmse, color=colors[model_idx], marker=markers[model_idx])

    plt.xlabel('Truth OD Flow Range')
    plt.ylabel('RMSE')
    plt.title('RMSE by Distribution of OD Flow')
    plt.legend(model_names)
    plt.savefig('./od_distribution.jpg')

def calculate_metrics(X, X_tilde):
    sum_squared_error = 0
    sum_absolute_percentage_error = 0
    sum_min = 0
    sum_actual_forecast = 0

    # Combine the keys from both dictionaries to make sure we cover all non-zero elements
    if len(X)>len(X_tilde):
        all_keys = X.keys()
    else:
        all_keys= X_tilde.keys()
    n_elements = len(all_keys)
    plot_sample = [[],[]]
    
    for key in all_keys:
        actual = X.get(key, 0)
        forecast = X_tilde.get(key, 0)
        diff = actual - forecast
        plot_sample[0].append(actual)
        plot_sample[1].append(diff)
        
        # RMSE components
        sum_squared_error += diff**2
        
        # SMAPE components, handling division by zero if both actual and forecast are zero
        denominator = (abs(actual) + abs(forecast))/ 2 + 1e-20
        if denominator > 0:
            sum_absolute_percentage_error += abs(diff) / denominator
        
        # CPC components
        sum_min += min(actual, forecast)
        sum_actual_forecast += actual + forecast
    
    # Final RMSE calculation
    rmse = (sum_squared_error / n_elements)**0.5
    
    # Final SMAPE calculation
    smape = (sum_absolute_percentage_error / n_elements)
    
    # Final CPC calculation, handling division by zero if sum_actual_forecast is zero
    cpc = (2 * sum_min) / sum_actual_forecast
    
    return rmse, smape, cpc, plot_sample

def get_od(Cityname,cell_size):
    edges = gpd.read_file(f'dataset/{Cityname}_data/map/edges.shp')

    # 计算边界
    points = edges['geometry'].centroid  # 注意这里交换了x和y的顺序
    points_meters = [(point.y * 111000, point.x * 111000) for point in edges['geometry'].centroid]
    min_x = min(points_meters, key=lambda x: x[0])[0]
    min_y = min(points_meters, key=lambda x: x[1])[1]

    # 加载数据集
    traj_test = np.load(f'dataset/{Cityname}_data/traj_test.npy')
    od_matrix = []
    for i in range(traj_test.shape[0]):
        tmp = traj_test[i]
        ori_index = tmp[0]  # 起始点索引
        dest_index = tmp[2]  # 终止点索引
        # 获取网格索引
        _, start_cell = cell_center(points[int(ori_index)], min_x, min_y, cell_size)
        _, end_cell = cell_center(points[int(dest_index)], min_x, min_y, cell_size)
        
        # 更新OD矩阵
        od_key = (start_cell, end_cell)
        od_matrix.append(od_key)

    return od_matrix

def calculate_curve_points(start, end, num_of_points=10, height_factor=0.02):
    # Vector from start to end
    vector = np.array(end) - np.array(start)
    # Perpendicular vector
    perp_vector = np.array([-vector[1], vector[0]])
    # Normalize the perpendicular vector
    perp_vector = height_factor * perp_vector / np.linalg.norm(perp_vector)
    # Calculate points along the curve
    curve_points = [start]
    for i in range(1, num_of_points + 1):
        fraction = i / (num_of_points + 1)
        point = (1 - fraction) * np.array(start) + fraction * np.array(end)
        height = np.sin(fraction * np.pi)  # Sin curve for smoothness
        curve_point = point + height * perp_vector
        # Convert numpy floats to native Python floats and check for NaNs
        if not np.any(np.isnan(curve_point)):
            curve_points.append([float(cp) for cp in curve_point])
    curve_points.append(end)
    return curve_points
def visualize_od_flow(truth, error, all_keys, centroids, map_center, boundaries, map_zoom_start=12):
    [boundary_start_min, boundary_start_max, boundary_end_min, boundary_end_max] = boundaries
    m = folium.Map(location=map_center, zoom_start=map_zoom_start)
    (x, y) = all_keys
    max_flow = max(truth)
    min_flow = min(truth)
    # 设置颜色映射
    disturbance = 0.001
    line_weight = 3
    cmap = plt.get_cmap('plasma')
    norm = mcolors.LogNorm(vmin=1, vmax=max_flow)
    
    data_to_save = []

    for i in tqdm(range(len(truth))):
        flow = truth[i] - error[i]
        start_idx = x[i]
        end_idx = y[i]
        if boundary_start_min <= start_idx <= boundary_start_max and boundary_end_min <= end_idx <= boundary_end_max and flow > 0:
            start_centroid = centroids[start_idx]
            end_centroid = centroids[end_idx]
            # Add start point with node ID
            folium.CircleMarker(
                location=[start_centroid[0], start_centroid[1]],
                color='green',
                radius=7,
                fill=True,
                fill_color='green',
                fill_opacity=1,
                popup=f'Start Node ID: {start_idx}'
            ).add_to(m)

            # Add end point with node ID
            folium.CircleMarker(
                location=[end_centroid[0] + disturbance, end_centroid[1] + disturbance],
                color='white',
                radius=7,
                fill=True,
                fill_color='white',
                fill_opacity=1,
                popup=f'End Node ID: {end_idx}'
            ).add_to(m)

            # Normalize flow for color intensity
            color = mcolors.to_hex(cmap(norm(flow)))

            locations = calculate_curve_points([start_centroid[0], start_centroid[1]], [end_centroid[0], end_centroid[1]], num_of_points=10, height_factor=0.03)
            
            folium.PolyLine(
                locations,
                color=color,
                weight=line_weight,
                popup=f'flow = {flow}, truth = {truth[i]}'
            ).add_to(m)
            
            # Append data for CSV
            data_to_save.append([start_idx, end_idx, flow])

    # Save data to CSV
    df = pd.DataFrame(data_to_save, columns=['cell_x', 'cell_y', 'predicted'])
    df.to_csv('od_flow_god.csv', index=False)

    colormap = cm.LinearColormap(
        colors=[mcolors.to_hex(cmap(norm(i))) for i in range(int(min_flow), int(max_flow) + 1)],
        vmin=min_flow,
        vmax=max_flow,
        caption='OD Flow'
    )
    colormap.add_to(m)
    return m

def cell_center_from_index(cell_index, min_x, min_y, cell_size):
    """
    Calculate the center point of the cell given its cell index.
    Assumes cell_index is a tuple (x_index, y_index).
    """
    # Approximate conversion factor from degrees to meters at Beijing's latitude
    conversion_factor = 111000

    # Adjust boundaries to be multiples of cell_size
    grid_min_x = min_x - (min_x % cell_size)
    grid_min_y = min_y - (min_y % cell_size)

    (x_index, y_index) = cell_index

    # Calculate the cell center in meters
    cell_x = x_index * cell_size + cell_size / 2 + grid_min_x
    cell_y = y_index * cell_size + cell_size / 2 + grid_min_y

    # Convert back to degrees
    return (cell_x / conversion_factor, cell_y / conversion_factor)

def get_spatio(opt, city):
    edges = gpd.read_file(f'dataset/{city}_data/map/edges.shp')
    cell_size = opt.cell_size
    points = [(point.y* 111000, point.x* 111000) for point in edges['geometry'].centroid]
    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]
    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    cell_df = pd.read_csv('./dataset/' + city + f'_data/cells_with_poi.csv')
    all_cells = cell_df['index'].apply(eval).to_list()
    centroids = [cell_center_from_index(cell, min_x, min_y, cell_size) for cell in all_cells]
    center = [(min_x + max_x) / (2* 111000), (min_y + max_y) / (2* 111000)]
    return centroids, center

if __name__ == '__main__':
    # one cell POI list :[0, 2 ,4]
    traj = np.zeros(750)
    for i in [0, 2 ,4]:
        traj[i,28] = 1

    # We could end up learning something
    traj[0, 28] = 1
    traj[2, 27] = 1
    traj[4, 26] = 1
    # It's hard to decide the distance
