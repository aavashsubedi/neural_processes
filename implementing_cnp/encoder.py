import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)

    def forward(self, context_x, context_y):

        encoder_data = torch.cat([context_x, context_y], dim=-1)

        #set size represents the number of context points, 
        #filter size is the number of features in the data
        batch_size, set_size, filter_size = encoder_data.size()

        x = encoder_data.view(batch_size * set_size, -1) # [B * N, 2] why would you do this? You shouldnt
        #you would do the x = data.view(batch_size * set_size, -1) so that you can 
        #do the fc1(x) and then do the x.view(batch_size, set_size, -1) to get the
        #output in the correct shape
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))

        represnetations = x.view(batch_size, set_size, -1)
        #so we keep [batch_size, -1]
        
        aggregated_representation = torch.mean(represnetations, dim=1) 

        return aggregated_representation
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()
        #the input is the shape of representation + shape of target_x
        self.fc1 = nn.Linear(129, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 2)
        
    def forward(self, representations, target_x):
        
        #target_x.shape = [batch_size, num_pred_points, dim_prediction]
        batch_size, filter_size, dim_pred = target_x.size()

        """
        we wamt to add the representation of each batch element to each target point i
        in the same batch element
        """ 
        representations = representations.unsqueeze(1).repeat(1, target_x.size(1), 1)

        x = torch.cat([representations, target_x], dim=-1) # [B, N, representation + shape of target_x]
        x = x.view(batch_size * target_x.size(1), -1) #so we can do the fc1(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(batch_size, filter_size, -1)

        mu, log_sigma = x.split(1, dim=-1)
        sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma) #softplus is a smooth relu
        distribution = torch.distributions.Normal(mu, sigma)
        return distribution, mu, sigma

class CNP(nn.Module):

    def __init__(self):
        super(CNP, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, context_x, context_y, target_x):
        #get the aggregated representation
        aggregated_representation = self.encoder(context_x, context_y)
        #get the distribution
        distribution, mu, sigma = self.decoder(aggregated_representation, target_x)

        return distribution, mu, sigma