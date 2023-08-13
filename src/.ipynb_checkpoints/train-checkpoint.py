from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from dataset import BaseDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from config import model_name
from tqdm import tqdm
import os
from pathlib import Path
from evaluate import evaluate
import importlib
import datetime
from config import model_name
import random 
import argparse
from utils.utils import latest_checkpoint


SEED_VALUE = 42
random.seed(SEED_VALUE)
torch.manual_seed(SEED_VALUE)
torch.cuda.manual_seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

try:
    Model = getattr(importlib.import_module(f"model.{model_name}"), model_name)
    config = getattr(importlib.import_module('config'), f"{model_name}Config")
    print(config)
except AttributeError:
    print(f"{model_name} not included!")
    exit()   
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.Inf

    def __call__(self, val_loss):
        """
        if you use other metrics where a higher value is better, e.g. accuracy,
        call this with its corresponding negative value
        """
        if val_loss < self.best_loss:
            early_stop = False
            get_better = True
            self.counter = 0
            self.best_loss = val_loss
        else:
            get_better = False
            self.counter += 1
            if self.counter >= self.patience:
                early_stop = True
            else:
                early_stop = False

        return early_stop, get_better

def train(data_folder_path):
    # writer = SummaryWriter(
    #     log_dir=
    #     f"./runs/{model_name}/{datetime.datetime.now().replace(microsecond=0).isoformat()}{'-' + os.environ['REMARK'] if 'REMARK' in os.environ else ''}"
    # )
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
   
    model = Model(config).to(device)
    
    print('Training path: ', os.path.join(data_folder_path, 'train/behaviors_data.csv'))
    dataset = BaseDataset(os.path.join(data_folder_path, 'train/behaviors_data.csv'), os.path.join(data_folder_path, 'segment_news_data.csv'))

    print(f"Load training dataset with size {len(dataset)}.")

    dataloader = DataLoader(dataset,
                   batch_size=config.batch_size,
                   shuffle=True)
                   # num_workers=config.num_workers,
                   # drop_last=True,
                   # pin_memory=True)
    
    ## optim, criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config.learning_rate)
    
    start_time = time.time()
    loss_full = []
    exhaustion_count = 0
    epoch = 0
    early_stopping = EarlyStopping()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_dir = os.path.join(f'./checkpoint/{model_name}', timestamp)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    checkpoint_path = latest_checkpoint(f'./checkpoint/{model_name}')
    if checkpoint_path is not None:
        print(f"Load saved parameters in {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        early_stopping(checkpoint['early_stop_value'])
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.train()

    for i in range(config.num_epochs):
        epoch += 1
        print(f'Epoch {epoch}')
        try:
            minibatch = next(iter(dataloader))
        except StopIteration:
            exhaustion_count += 1
            tqdm.write(
                f"Training data exhausted for {exhaustion_count} times after {i} batches, reuse the dataset."
            )
            dataloader = iter(
                DataLoader(dataset,
                           batch_size=config.batch_size,
                           shuffle=True))
                           # num_workers=config.num_workers,
                           # drop_last=True,
                           # pin_memory=True)
                
            minibatch = next(dataloader)
        for step, minibatch in enumerate(tqdm(dataloader)):

            y_pred = model(minibatch["candidate_news"],
                           minibatch["clicked_news"])
#             print(minibatch['clicked'])
            # y =torch.stack(minibatch['clicked']).float().transpose(0, 1).to(device)
#             print(y.shape)
            # print(y)
            y = minibatch['clicked'].float().to(device)
#             y = torch.zeros(len(y_pred)).long().to(device)
            loss = criterion(y_pred, y)

            loss_full.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # if i % 10 == 0:
        #     writer.add_scalar('Train/Loss', loss.item(), step)

            if step % config.num_steps_show_loss == 0:
                tqdm.write(
                    f"Time {time_since(start_time)}, batches {step}, current loss {loss.item():.4f}, average loss: {np.mean(loss_full):.4f}, latest average loss: {np.mean(loss_full[-256:]):.4f}"
                )
            
        print('Evaluation !!!')
        # if i % config.num_batches_validate == 0:
        model.eval()
        val_auc, val_mrr, val_ndcg5, val_ndcg10 = evaluate(
            model, data_folder_path,
            config.num_workers, 200000)
        model.train()
        tqdm.write(
            f"Time {time_since(start_time)}, epoch {epoch}, validation AUC: {val_auc:.4f}, validation MRR: {val_mrr:.4f}, validation nDCG@5: {val_ndcg5:.4f}, validation nDCG@10: {val_ndcg10:.4f}, "
        )

        early_stop, get_better = early_stopping(-val_auc)
        if early_stop:
            tqdm.write('Early stop.')
            break
        elif get_better:
            try:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'epoch':epoch,
                        'early_stop_value':
                        -val_auc
                    }, f"{checkpoint_dir}/ckpt-{epoch}.pth")
            except OSError as error:
                print(f"OS error: {error}")

def time_since(since):
    """
    Format elapsed time string.
    """
    now = time.time()
    elapsed_time = now - since
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


if __name__ == '__main__':
    print('Using device:', device)
    print(f'Training model {model_name}')
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_folder_path', type=str, help = 'path to the data fold')
    
    args = parser.parse_args()
    
    data_folder_path = args.data_folder_path
    
    train(data_folder_path)
