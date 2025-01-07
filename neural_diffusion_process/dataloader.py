import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import datasets


"""
We need to get batch.mask_data somehow
"""
def get_image_grid_inputs(size: int, a=2): #map to a [-2, 2] grid
    x1 = np.linspace(-a, a, size)
    x2 = np.linspace(-a, a, size)
    x1, x2 = np.meshgrid(x1, x2)
    return np.stack([x1.ravel(), x2.ravel()], axis=-1) #what does ravel do?
    #ravel flattens the array


def get_rescale_func_fwd_inv(dataset_name="mnist"):
    
    mean = torch.zeros(1).view(1, 1)
    std = torch.ones(1).view(1, 1)

    def fwd(y):
        return y
    def inv(y):
        y = y * std + mean
        y = torch.clamp(y, min=0, max=1)
        return y
    return fwd, inv

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, input_grid, rescale_func):
        self.dataset = dataset
        self.input_grid = input_grid
        #self.rescale_func = rescale_func

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):

        img, label = self.dataset[idx]
        img = img.view(-1, img.shape[0]) #Reshape to [H * W, C]
        y_values = img

        return self.input_grid, y_values

def get_data(image_size=28, dataset_name="mnist", path="dataset/mnist", 
    batch_size=32, num_epochs=1, bw=True):

    input_grid = get_image_grid_inputs(image_size)
    rescale, _ = get_rescale_func_fwd_inv(dataset_name)

    # if bw:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        torchvision.transforms.Grayscale(num_output_channels=1),
        #scale the data to [-2, 2]        
    ])

    dataset = torchvision.datasets.ImageFolder(
        path, transform=transform
    )
    wrapped_dataset = CustomDataset(dataset, input_grid, rescale)
    #current dataset is between [0, 1]

    dataloader = DataLoader(wrapped_dataset, batch_size=batch_size, shuffle=True,
                            )
                            
    # def preprocess(batch):
    #     images = batch[0]
    #     batch_size = images.shape[0]
    #     x = torch.tensor(input_grid)
    #     y = images.view(batch_size, -1, image.size(1))
    #     y = rescale(y)
    #     return dict(
    #         x_target=x,
    #         y_target=y
    #     )
    return dataloader #maybe we can preprocess this alter.

def get_context_mask(image_size=28, 
                    context_typ: str="percent",
                    p:float=0.5,
                    KEEP=33, NOT_KEEP=44):

    #KEEP/NOT_KEEP are random values
    """
    Generates a context mask for a given image size and 
    context type.

    Args:
    - image_size: int, the size of the image
    - context_typ: str, the type of context mask to generate
    - p: float, the probability of masking a pixel
    """

    x = torch.tensor(get_image_grid_inputs(image_size)) #[h*w, 2]
    
    if context_type == "horizontal":
        mask = x[:, 1] > 0.0 #mask the right half of the image
    elif context_type == "vertical":
        mask = x[:, 0]  < 0.0 #mask the top half of the image

    elif "percent" in context_type:
        contition = torch.rand(len(x)) < p

    mask = torch.where(
        condition,
        KEEP * torch.ones_like(x[..., 0]),
        NOT_KEEP * torch.ones_like(x[..., 0])
    )
