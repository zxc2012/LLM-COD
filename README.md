## [Harnessing LLMs for Cross-City OD Flow Prediction(Sigspatial 2024)](https://dl.acm.org/doi/10.1145/3678717.3691308)

## Package Dependencies

The project will also need following package dependencies:
- transformers
- bitsandbytes
- peft

## Pretrained checkpoints

A100 weights could be downloaded at [link]()

## Format of the Data
```python
  ori_id ori_time dest_id  dest_dist
0   酒店宾馆    01:57      购物       2.04
1   产业园区    16:42    生活服务       5.71
2   公司企业    13:31     住宅区      13.36
3   医疗保健    01:02    教育学校       4.48
4   运动健身    02:19    运动健身       1.83
```

We obtained the origin Beijing, Chengdu and Xi'an Data from a [VLDB paper](https://connectpolyu-my.sharepoint.com/:f:/g/personal/21037065r_connect_polyu_hk/EgvyOyo1eWNEjPcSjSsVM-0BQGVrfuA0NdTV8ocg6QsaJA?e=gGCXCf)

### OSM map data
In the `dataset/city/map` folder, there are the following files:

1. `nodes.shp` : Contains OSM node information with unique `node id`.
2. `edges.shp` : Contains network connectivity information with unique `edge id`.

### POI data

The origin POI data was obtained through tencent map API in preprocess/poi.py

## Usage

- `nohup python3 -u main.py --city bj > logs/log.txt 2>&1 &`

