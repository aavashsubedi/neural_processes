import torch
import torch.nn as nn
import torch.optim as optim
import collections
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from dataloader import GPCurvesReader, CNPRegressionDescription
from encoder import Encoder, Decoder, CNP


def train():

    dataset_train = GPCurvesReader(
        batch_size=1280, max_num_context=10, x_size=1, y_size=1
    )

    dataset_test = GPCurvesReader(
        batch_size=128, max_num_context=30, x_size=1, y_size=1, testing=True
    )

    d_x, d_in, representation_size, d_out = 1, 2, 128, 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    import pdb; pdb.set_trace()
    for epoch in range(100):
            
            model.train()
            total_loss = 0
            for i in range(50):
                optimizer.zero_grad()
                data = dataset_train.generate_curves()
                (context_x, context_y), target_x = data.query
                target_y = data.target_y
    
                context_x = context_x.to(device)
                context_y = context_y.to(device)
                target_x = target_x.to(device)
                target_y = target_y.to(device)

                dist, mu, sigma = model(context_x, context_y, target_x)
                log_p = dist.log_prob(target_y)
                loss = -log_p.mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch: {epoch}, Loss: {total_loss / 1000}")
            model.eval()
            with torch.no_grad():
                data = dataset_test.generate_curves()
                (context_x, context_y), target_x = data.query
                target_y = data.target_y

                context_x = context_x.to(device)
                context_y = context_y.to(device)
                target_x = target_x.to(device)
                target_y = target_y.to(device)
                dist, mu, sigma = model(context_x, context_y, target_x)
                log_p = dist.log_prob(target_y)
                loss = -log_p.mean()
                print(f"Test Loss: {loss.item()}")
    
    
 
               # Forward pass
    
if __name__ == "__main__":
    train()