import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class DeterministicEncoder(nn.Module):

    def __init__(self):
        super(DeterministicEncoder, self).__init__()
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
        import pdb; pdb.set_trace()
        aggregated_representation = torch.mean(represnetations, dim=1) 

        return aggregated_representation


class LatentEncoder(nn.Module):
    """
    Takes a represnetation and computes the mean and sigma of the latent representation
    """
    def __init__(self):
        
        super(LatentEncoder, self).__init__()
        #the input is the shape of representation
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        #self.penultiamte layer needed her that outputs the shape
        #(last_layer_size + self.num_latets) / 2
        self.penultimate = nn.Linear(128, 128)

        #second shape is the input to the decoder. Or the number of latents
        self.mu_linear = nn.Linear(128, 128)
        self.sigma_linear = nn.Linear(128, 128)

    def forward(self, x, y):

        encoder_data = torch.cat([x, y], dim=-1)

        #set size represents the number of context points, 
        #filter size is the number of features in the data
        batch_size, set_size, filter_size = encoder_data.size()

        x = encoder_data.view(batch_size * set_size, -1) 
        x = torch.relu(self.fc1(encoder_data))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = x.view(batch_size, set_size, -1)
        #take mean over samples
        x = torch.mean(x, dim=1) #mean across all contexts, retuns shape [batch_size, -1]
        x = torch.relu(self.penultimate(x))
        mu = self.mu_linear(x)
        log_sigma = self.sigma_linear(x)
        
        sigma = 0.1 + 0.9 * nn.functional.softplus(log_sigma) #softplus is a smooth relu
        distribution = torch.distributions.Normal(mu, sigma)
        
        return distribution 

class LatentOnlyDecoder(nn.Module):
    """
    Decode the latent representations, 
    """
    def __init__(self):
        super(LatentOnlyDecoder, self).__init__()
        #the input is the shape of representation + shape of target_x
        self.fc1 = nn.Linear(129, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        #output is the dim_prediction * 2 (for mean and variance).
        self.fc4 = nn.Linear(128, 2)
        
    def forward(self, representations, target_x):
        """
        representations are of shape: [batch_size, num_context_points, dim_representation]
        """
        #target_x.shape = [batch_size, num_pred_points, dim_prediction]
        batch_size, filter_size, dim_pred = target_x.size()

        """
        we wamt to add the representation of each batch element to each target point i
        in the same batch element
        """ 

        #concetanate the latent representation z to the target_x
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


class LNP(nn.Module):
    """
    Returns:
    log_p: log probability of the target_y given the distribution. Shape = [B, num_targets]
    mu: mean of the distribution. Shape = [B, num_targets, d_y]
    sigma: std of the distribution. Shape = [B, num_targets, d_y]
    """
    def __init__(self):
        super(LNP, self).__init__()
        self.encoder = DeterministicEncoder()
        self.latent_encoder = LatentEncoder()
        self.decoder = LatentOnlyDecoder()

    def forward(self, context_x, context_y, target_x, target_y=None):
        #get the aggregated representation
        prior_dist = self.latent_encoder(context_x, context_y)


        if target_y is None:
            latent_representation = prior_dist.sample()
        else:
            posterior = self.latent_encoder(target_x, target_y)
            latent_representation = posterior.sample()

        latent_representation = latent_representation.unsqueeze(1).repeat(1,
         target_x.size(1), 1)
        #deocderf expects shape of [batch_size, num_pred_points, dim_prediction] and outputs 
        #a distribution

        #get the distribution
        distribution, mu, sigma = self.decoder(latent_representation, target_x)
         #computes the log probability of the target_y given the distribution

        if target_y is not None:
            log_p = distribution.log_prob(target_y).squeeze(-1) #log_p has shape [batch_size, num_pred_points, dim_prediction]
            posterior = self.latent_encoder(target_x, target_y)
            #kl div has shape [batch_size, num_pred_points, dim_prediction]

            #not sure about this part?
            kl = torch.distributions.kl.kl_divergence(posterior, prior_dist).sum(axis=-1, keepdim=True) 
            #repeat kl for each target based on the batch size
            loss = -torch.mean(log_p - kl / target_x.size(1))
        else:
            log_p, kl, loss = None, None, None
        
        return mu, sigma, log_p, kl, loss

