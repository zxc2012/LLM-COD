import time
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import geopandas as gpd
from sklearn.model_selection import GridSearchCV
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from experiment import calculate_metrics, get_spatio, plot_rmse_by_od_flow, visualize_od_flow
from args import make_args


opt = make_args()
print(opt)


def random_forest(train_x, train_y, x, y, test_x_tar, tar_index_tar, test_y_tar):
    # 定义参数网格
    param_grid = {
        'n_estimators': [200],
        'max_depth': [20],
        'min_samples_split': [2],
        'min_samples_leaf': [4]
    }

    # 创建随机森林模型
    rf = RandomForestRegressor()

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)

    # 拟合网格搜索
    grid_search.fit(train_x, train_y)
    # 查看最优参数和最优分数
    print("Best parameters:", grid_search.best_params_)
    print("Best score:", grid_search.best_score_)

    # 使用最佳模型进行预测
    best_model = grid_search.best_estimator_

    predictions_tar = best_model.predict(test_x_tar)
    predictions = {idx: tar for idx, tar in zip(zip(x, y), predictions_tar)}
    truth = {idx: tar for idx, tar in zip(zip(tar_index_tar[0], tar_index_tar[1]), test_y_tar)}
    # 计算并打印评估指标
    rmse, smape, cpc, plot_sample = calculate_metrics(truth, predictions)
    all_keys = predictions.keys()

    print('After Fine-tuning - Target City Test RMSE =', rmse)
    print('After Fine-tuning - Target City Test SMAPE =', smape)
    print('After Fine-tuning - Target City Test CPC =', cpc)
    return plot_sample, all_keys

def gravity(train_x, train_y, x, y, test_x_tar, tar_index_tar, test_y_tar):
    train_x = torch.FloatTensor(np.log(train_x + 1e-20))
    train_y = torch.FloatTensor(np.log(train_y))
    # 模型设定
    class Model(torch.nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.linear1 = nn.Linear(train_x.shape[1], 1)  # 自动调整输入维度

        def forward(self, x):
            x = self.linear1(x)
            return x.squeeze()
        
    # 初始化模型和优化器
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.MSELoss()

    # 模型训练过程
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset=dataset, batch_size=2000, shuffle=True)
    for epoch in range(100):  # 训练轮数
        model.train()
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

    test_x_new = torch.FloatTensor(np.log(test_x_tar + 1e-20))
    test_y_new = torch.FloatTensor(test_y_tar)
    # 使用模型进行新城市数据的预测
    model.eval()
    with torch.no_grad():
        predictions_new = model(test_x_new.to(device)).cpu().numpy()
        predictions_new = np.exp(predictions_new).reshape(-1)

    predictions = {idx: tar for idx, tar in zip(zip(x, y), predictions_new)}
    truth = {idx: tar for idx, tar in zip(zip(tar_index_tar[0], tar_index_tar[1]), test_y_tar)}
    # print(predictions, truth)

    rmse, smape, cpc, plot_sample = calculate_metrics(truth, predictions)
    print('After Fine-tuning - Target City Test RMSE =', rmse)
    print('After Fine-tuning - Target City Test SMAPE =', smape)
    print('After Fine-tuning - Target City Test CPC =', cpc)
    return plot_sample

def GBRT(train_x, train_y, x, y, test_x_tar, tar_index_tar, test_y_tar):
    # 定义参数网格
    param_grid = {
        'n_estimators': [200],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [2]
    }

    # 创建 Gradient Boosting 模型
    gbrt = GradientBoostingRegressor()

    # 创建网格搜索对象
    grid_search = GridSearchCV(estimator=gbrt, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

    # 执行网格搜索
    start = time.time()
    grid_search.fit(train_x, train_y)
    end = time.time()

    # 输出最佳参数和最佳模型的分数
    print('Best parameters:', grid_search.best_params_)
    print('Best score (MSE):', grid_search.best_score_)
    print('Time consumed for grid search: {:.2f} seconds'.format(end - start))
    # 使用最佳模型预测
    best_model = grid_search.best_estimator_

    predictions_tar = best_model.predict(test_x_tar)
    predictions = {idx: tar for idx, tar in zip(zip(x, y), predictions_tar)}
    truth = {idx: tar for idx, tar in zip(zip(tar_index_tar[0], tar_index_tar[1]), test_y_tar)}
    # print(predictions, truth)

    rmse, smape, cpc, plot_sample = calculate_metrics(truth, predictions)


    # 计算目标城市的评估指标
    print('Target City Test RMSE =', rmse)
    print('Target City Test SMAPE =', smape)
    print('Target City Test CPC =', cpc)
    return plot_sample

if __name__ == '__main__':
    # 加载训练城市数据
    city = 'bj'
    od = np.load(f'./GOD/data/od/{city}/train.npy')
    data = pd.read_csv(f'./dataset/{city}_data/poi_feature_{city}_{opt.cell_size}.csv')
    feat = data.values
    dis = np.load(f'./GOD/data/adj/{city}/distance.npy')
    train_index = pickle.load(open(f'./GOD/data/od/{city}/train_index.pkl', 'rb'))
    test_index = pickle.load(open(f'./GOD/data/od/{city}/test_index.pkl', 'rb'))

    # 准备训练和测试数据
    train_dis = dis[train_index].reshape([-1, 1])
    train_x = np.concatenate((feat[train_index[0]], feat[train_index[1]], train_dis), axis=1)
    train_y = od[train_index]
    test_dis = dis[test_index].reshape([-1, 1])
    test_x = np.concatenate((feat[test_index[0]], feat[test_index[1]], test_dis), axis=1)
    test_y = od[test_index]

    # 假设目标城市的标识
    tar_city = 'cd'

    # 加载目标城市的数据
    od_tar = np.load(f'./GOD/data/od/{tar_city}/test.npy')
    data_tar = pd.read_csv(f'./dataset/{tar_city}_data/poi_feature_{tar_city}_{opt.cell_size}.csv')
    feat_tar = data_tar.values
    dis_tar = np.load(f'./GOD/data/adj/{tar_city}/distance.npy')
    tar_index_tar = pickle.load(open(f'./GOD/data/od/{tar_city}/test_index.pkl', 'rb'))


    cell_df = pd.read_csv('./dataset/' + tar_city + f'_data/cells_with_poi.csv')
    cell_df = cell_df.dropna(subset=['POI'])
    mapping = pd.read_csv(f'./dataset/{tar_city}_data/node_mapping.csv')
    mapping_dict = pd.Series(mapping.node_id.values, index=mapping.cell).to_dict()


    cell_df['index'] = cell_df['index'].map(lambda x: mapping_dict.get(x))
    all_cells = cell_df['index'].to_list()

    x = [single for i in range(len(all_cells)) for single in all_cells] # [All_cells, All_cells]
    y = [single for single in all_cells for i in range(len(all_cells))] # [1,1,1,1...]

    test_index_tar = (x, y)
    # 准备目标城市的测试数据
    test_dis_tar = dis_tar[test_index_tar].reshape([-1, 1])
    test_x_tar = np.concatenate((feat_tar[x], feat_tar[y], test_dis_tar), axis=1)
    test_y_tar = od_tar[tar_index_tar]

    rf_sample, all_keys = random_forest(train_x, train_y, x, y, test_x_tar, tar_index_tar, test_y_tar)
    GM_sample = gravity(train_x, train_y, x, y, test_x_tar, tar_index_tar, test_y_tar)
    GBRT_sample = GBRT(train_x, train_y, x, y, test_x_tar, tar_index_tar, test_y_tar)
    GOD_sample = np.load('./GOD.npy')
    our_sample = np.load('./ours.npy')
    # print("error_sum:", np.sum(our_sample))
    
    # print("The errors of our model for each OD flow:", our_sample)
    # print("The erors of GOD for each OD flow:", GOD_sample)

    model_names = ['random_forest', 'gravity', 'GBRT', 'GOD', 'Ours']
    testing_model = [rf_sample[1], GM_sample[1], GBRT_sample[1], GOD_sample, our_sample]

    # model_names = ['GOD', 'Ours']
    # testing_model = [GOD_sample, our_sample]

    print(len(rf_sample[0]), [len(model) for model in testing_model])
    plot_rmse_by_od_flow([rf_sample[0]] + testing_model, model_names)
    


    # Visualization
    centroids, map_center = get_spatio(opt, tar_city)
    x_vis = [x[i] for i in range(len(x)) if rf_sample[0][i] > 0]
    y_vis = [y[i] for i in range(len(x)) if rf_sample[0][i] > 0]
    boundary_start_min, boundary_start_max = min(x_vis), max(x_vis)
    boundary_end_min, boundary_end_max = min(y_vis), max(y_vis)
    od_map = visualize_od_flow(rf_sample[0], [0]*len(rf_sample[0]), test_index_tar, centroids, map_center, [boundary_start_min, boundary_start_max, boundary_end_min, boundary_end_max])
    od_map.save(f'./results/Road_HTML/{opt.city}_odflow_ground.html')
    for idx, error in enumerate(testing_model):
        od_map = visualize_od_flow(rf_sample[0], error, test_index_tar, centroids, map_center, [boundary_start_min, boundary_start_max, boundary_end_min, boundary_end_max])
        od_map.save(f'./results/Road_HTML/{opt.city}_odflow_{model_names[idx]}.html')

