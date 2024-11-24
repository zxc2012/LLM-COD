from args import make_args
from preprocess import generateEdgesByCity, gentrajByCity, prepare_instruction
from train import train,test
import pandas as pd

  
if __name__ == '__main__':
    opt = make_args()
    print(opt)

    # 1 Prepare edges and traj
    # for city in ['bj','cd','xa']:
    #     # generateEdgesByCity(city)
    #     gentrajByCity(city, cell_size = opt.cell_size)
    
    gentrajByCity(opt.city, cell_size = opt.cell_size)
    # 2 Generate Split Data and analysis
    train_dataset = prepare_instruction(opt,type='train')
    print(train_dataset.head())
    valid_dataset = prepare_instruction(opt,type='valid')
    test_dataset = prepare_instruction(opt,type = 'test')
    # visualize_dataset(train_dataset)
    # train(train_dataset,valid_dataset,opt)
    test(test_dataset,opt)

    ## Running experiment

    
