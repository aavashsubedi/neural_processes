import torch
import torch.nn as nn
from tqdm import tqdm 

class EMA(nn.Module):

	def __init__(self, beta=0.999):
		super(EMA, self).__init__()
		self.beta = beta
		self.step = 0
	def update_model_average(self, ema_model, model):

		for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = old_weight * self.beta + new_weight * (1 - self.beta)

	def update_average(self, old, new):
		return old * self.beta + new * (1 - self.beta)
	
	def reset_parameters(self, ema_model, model):
		#resets current model
		ema_model.load_state_dict(model.state_dict())

	def step_ema(self, model, ema_model, step_start_ema=1000):
		"""
		step_start_ema: int, the step to start updating the ema model
		"""
		if self.step < step_start_ema:
			self.reset_parameters(ema_model, model)
		    self.step += 1
			return 0
		self.update_model_average(ema_model, model)
		self.step += 1

