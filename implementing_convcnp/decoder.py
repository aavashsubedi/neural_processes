import torch 
import torch.nn as nn


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
        x = x.view(batch_size, target_x.size(1), -1)
        return xq
