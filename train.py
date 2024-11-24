from datasets import Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
)
from experiment import find_closest_vector, get_od, calculate_metrics, plot_rmse_by_od_flow
from preprocess import get_poi,get_bins
import torch
import pandas as pd
import numpy as np
from conf import CustomTrainer
from peft import LoraConfig, get_peft_model
from evaluate import evaluator
from sklearn.metrics import precision_score, recall_score, f1_score, label_ranking_average_precision_score

def precision_at_k(y_true, y_pred, k):
    y_true = y_true.float()
    y_pred = y_pred.float()
    # Assuming y_true and y_pred are the same shape [batch_size, num_classes]
    # and contain probabilities or scores.

    # Get the indices of the top k predictions
    _, indices = y_pred.topk(k, dim=1, largest=True, sorted=True)

    # Create a tensor to gather the true labels of the top k predictions
    pred_labels = torch.zeros_like(y_pred).scatter_(1, indices, 1)

    # Calculate precision
    correct = (pred_labels * y_true).sum(dim=1)
    precision = correct / k

    return precision.mean()

def formatting_func(example, city):
    output_texts = []
    for i in range(len(example['ori_id'])):
        text = f"""Given the starting place and time of a taxi trajectory in {city}, predict the most likely destination 
        Starting place:{example['ori_id'][i]}, Starting time:{example['ori_time'][i]}
        """
        # text = f"""Given the starting place and time of a taxi trajectory, predict the most likely destination 
        # Starting place:{example['ori_id'][i]}, Starting time:{example['ori_time'][i]}
        # """
        output_texts.append(text)
    return output_texts

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Apply sigmoid to logits and threshold to convert to binary predictions
    predictions = (logits > 0.7)
    print(predictions.shape, labels.shape)
    # Calculate metrics
    precision = precision_score(labels, predictions, average='micro')
    recall = recall_score(labels, predictions, average='micro')
    f1 = f1_score(labels, predictions, average='micro')
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def multi_hot_encode_pois(dest_id_list, label2id):
    """Convert a list of POIs into a multi-hot encoded vector based on label2id mapping."""
    pois = dest_id_list['dest_id']
    vector = np.zeros(len(label2id), dtype=float)
    if type(pois) == list:
        for poi in pois:
            if poi in label2id:
                vector[label2id[poi]] = 1
                # vector[label2id[poi]] = dest_id_list['dest_dist']
    else:
        vector[label2id[pois]] = 1
        # vector[label2id[pois]] = dest_id_list['dest_dist']
    return vector.tolist()

def multi_hot_encode_cell(dest_id_list, label2id):
    pois = dest_id_list
    vector = np.zeros(len(label2id), dtype=float)
    if type(pois) == list:
        for poi in pois:
            if poi in label2id:
                vector[label2id[poi]] = 1
    else:
        vector[label2id[pois]] = 1
    return vector.tolist()

def train(traj_train_df,traj_valid_df,args):
    # Specify the name of the base model to load from Hugging Face model hub
    base_model = "NousResearch/Llama-2-7b-hf"
    # base_model = "google/gemma-2b"

    unique_poi_types = ['商务楼宇', '机构团体', '基础设施', '产业园区', '美食', '购物', '旅游景点', '酒店宾馆', '娱乐休闲', '文化场馆', '生活服务',
                        '汽车','住宅区', '公司企业', '教育学校', '医疗保健', '运动健身', '银行金融', '地名地址', '房产小区附属',
                        '其它房产小区', '室内及附属设施', '其它'
                        ]

    label2id = {poi: index for index, poi in enumerate(unique_poi_types)}
    traj_train_df['text'] = formatting_func(traj_train_df, args.city)
    traj_valid_df['text'] = formatting_func(traj_valid_df, args.city)

    traj_train_df['label'] = [multi_hot_encode_pois(dest_id_list, label2id) for _,dest_id_list in traj_train_df.iterrows()]
    traj_valid_df['label'] = [multi_hot_encode_pois(dest_id_list, label2id) for _,dest_id_list in traj_valid_df.iterrows()]

    traj_train_df = traj_train_df.drop(columns=['ori_id', 'ori_time', 'dest_id','dest_dist'])
    traj_valid_df = traj_valid_df.drop(columns=['ori_id', 'ori_time', 'dest_id','dest_dist'])
    
    traj_valid_df = Dataset.from_pandas(traj_valid_df)
    traj_train_df = Dataset.from_pandas(traj_train_df)


    # Set quantization configuration to reduce model size and improve inference speed
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )

    # Load the model from pre-trained, apply quantization configuration
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels= len(unique_poi_types),
        quantization_config=quant_config,
        device_map="auto",
        cache_dir = "./models/",
        problem_type="multi_label_classification" if args.use_multiple else "single_label_classification"
    )
    # Disable model cache to save memory, set parallel training parameters
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True,cache_dir = "./models/")
    tokenizer.pad_token = tokenizer.eos_token
    def preprocess_function(examples):
        return tokenizer(examples["text"], max_length=60, truncation=True,padding= "max_length")
    
    traj_train_df = traj_train_df.map(preprocess_function, batched=True)
    traj_valid_df = traj_valid_df.map(preprocess_function, batched=True)

    # Set PEFT (Parameter-Efficient Fine-Tuning) configuration
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_params)
    # Configure training parameters
    training_params = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=140,
        gradient_accumulation_steps=4,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=200,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine"
    )

    # Initialize and configure the trainer
    trainer = CustomTrainer(
        model= model,
        train_dataset= traj_train_df,
        eval_dataset = traj_valid_df,
        tokenizer= tokenizer,
        args= training_params,
        compute_metrics= compute_metrics,
    )

    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.model.save_pretrained('./results')
    trainer.tokenizer.save_pretrained('./results')

def test(traj_test_df, args):
    traj_test_df = traj_test_df
    cell_df = pd.read_csv('dataset/' + args.city + f'_data/cells_with_poi.csv')
    unique_poi_types = ['商务楼宇', '机构团体', '基础设施', '产业园区', '美食', '购物', '旅游景点', '酒店宾馆', '娱乐休闲', '文化场馆', '生活服务',
                        '汽车','住宅区', '公司企业', '教育学校', '医疗保健', '运动健身', '银行金融', '地名地址', '房产小区附属',
                        '其它房产小区', '室内及附属设施', '其它'
                        ]
    cell_df['POI'] = cell_df['POI'].apply(get_poi,use_multiple = True)
    cell_df = cell_df.dropna(subset=['POI'])

    label2id = {poi: index for index, poi in enumerate(unique_poi_types)}
    traj_test_df['text'] = formatting_func(traj_test_df, args.city)

    traj_test_df['label'] = [multi_hot_encode_pois(dest_id_list, label2id) for _,dest_id_list in traj_test_df.iterrows()]
    cell_df['POI'] = [multi_hot_encode_cell(dest_id_list, label2id) for dest_id_list in cell_df['POI']]

    cell_dict = pd.Series(cell_df['POI'].values, index=cell_df['index']).to_dict()

    traj_test_df = traj_test_df.drop(columns=['ori_id', 'ori_time', 'dest_id','dest_dist'])
    
    traj_test_df = Dataset.from_pandas(traj_test_df)

    model_path = "./results/"
    # model_path = "NousResearch/Llama-2-7b-hf"
    # Set quantization configuration to reduce model size and improve inference speed
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=getattr(torch, "float16"),
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels= len(unique_poi_types),
        quantization_config=quant_config,
        device_map="auto",
        cache_dir = "./models/",
        problem_type="multi_label_classification" if args.use_multiple else "single_label_classification"
    )
    # Set PEFT (Parameter-Efficient Fine-Tuning) configuration
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_params)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True,cache_dir = "./results/")
    tokenizer.pad_token = tokenizer.eos_token
    def preprocess_function(examples):
        output = tokenizer(examples["text"], max_length=60, truncation=True,padding= "max_length", return_tensors="pt")
        return {k: v.to('cuda') for k, v in output.items()}

    traj_test_df = traj_test_df.map(preprocess_function, batched=True)
    traj_test_df.set_format(type='torch')
    test_dataset = torch.utils.data.DataLoader(traj_test_df, batch_size=args.test_batch_size)

    od_sequence = get_od(args.city,args.cell_size)
    all_cells = cell_df['index'].apply(eval).to_list()

    x = [single for i in range(len(all_cells)) for single in all_cells] 
    y = [single for single in all_cells for i in range(len(all_cells))] 
    truth_matrix = {}
    for i in range(len(x)):
        truth_matrix[(x[i], y[i])] = 0
    predict_matrix = {}

    for (batch_index,batch) in enumerate(test_dataset):
        with torch.no_grad():
            outputs = model(input_ids = batch['input_ids'].to('cuda'))
            print(batch_index)
            for i in range(len(batch['label'])):
                start_cell, end_cell = od_sequence[batch_index*args.test_batch_size + i]
                od_key = (start_cell, end_cell)
                if od_key not in truth_matrix:
                    continue
                else:
                    truth_matrix[od_key] += 1
                closet_cell = find_closest_vector(outputs.logits[i], cell_dict, end_cell, alpha = 14.5)
                # if closet_cell != end_cell:
                #     print(start_cell, end_cell, closet_cell)
                # print(closet_cell, end_cell)
                # 更新OD矩阵
                predict_key = (start_cell, closet_cell)
                if predict_key not in predict_matrix:
                    predict_matrix[predict_key] = 1
                else:
                    predict_matrix[predict_key] += 1




    # precision = precision_score(labels.cpu(), logits.cpu(), average='micro')
    # recall = recall_score(labels.cpu(), logits.cpu(), average='micro')
    # f1 = f1_score(labels.cpu(), logits.cpu(), average='micro')
    # topk = precision_at_k(labels.cpu(), rankings.cpu(), k =3 ).item()

    rmse, smape, cpc, plot_sample = calculate_metrics(truth_matrix, predict_matrix)
    np.save('./ours.npy',plot_sample[1])
    result = {
        "rmse": rmse,
        "smape": smape,
        "cpc": cpc
    }
    print(result)