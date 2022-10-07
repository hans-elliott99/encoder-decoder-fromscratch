import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math

@torch.no_grad()
def viz_tensors_bars(tensor_list : list, names : list=None, y_ax_max="max_average", type:str=''):
    '''
    Plots tensors in layer-order as a bar-chart. Plots max and mean value of each tensor.    

    tensor_list -- the list of raw model parameters (either gradients or weights).\n
    names -- Optionally provide list of names matching the param order.\n  
    y_ax_max -- y axis limit and is either "max_average" (max of average 'type') or "mean_average" (mean of average 'type')\n
    type -- specify the type of parameter, for example 'gradient' or 'weight', to adjust plot labels\n
    
    Adapted from source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in enumerate(tensor_list):
        layers.append(str(n))
        ave_grads.append(p.abs().mean().item())
        max_grads.append(p.abs().max().item())

    xticks = range(0,len(ave_grads), 1)
    y_min = min(ave_grads)
    y_max = max(ave_grads)
    if y_ax_max == "mean_average":
        y_max = (sum(ave_grads)/len(ave_grads))
    if names is not None:
        for x_ix in xticks:
            plt.text(x_ix, y_max/3, s=names[x_ix], rotation=90)

    plt.bar(range(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(range(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(xticks, layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = y_min, top=y_max) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel(f"average {type.lower()}")
    plt.title(f"{type.capitalize()} Summary")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], [f'max-{type.lower()}', f'mean-{type.lower()}', f'zero-{type.lower()}'])



@torch.no_grad()
def viz_tensors_density(tensor_list:list, names:list, exclude:list=['bias'], print_info=True, title=''):
  """
  Plot gradients to look for vanishing or exploding gradients.  
  Plot weights to examine distribution throughout the network.
  """
  legends = []
  for i, (name, grad) in enumerate(zip(names, tensor_list)): # note: exclude the output layer
    if any(exc in name.split('_') for exc in exclude):
      pass
    else:
      if print_info:
        print('layer %d (%10s): mean %+f, std %e' % (i, name, grad.mean(), grad.std()))
      hy, hx = torch.histogram(grad, density=True)
      plt.plot(hx[:-1].detach(), hy.detach(), alpha=0.7) ##remove last x
      legends.append(f'layer {i} ({name})')
  plt.legend(legends);
  plt.title(title)



class UpdateRatioTracker():
    def __init__(self, params, names, total_iters, device, metric="std"):
        "Track the epoch-average weight update ratio for each layer"
        self.params = params
        self.names = names
        self.total_iters = total_iters
        self.metric = metric
        self.device = device

        self.counter = 0
        self.update_running = torch.zeros((total_iters, len(params)), device=device)

        self.output = {name:[] for name in names}

    def step(self, lr):
        """Include after every backprop step"""
        if self.metric=='std':
            self.update_running[self.counter] += torch.tensor([((lr*p.grad).std() / p.data.std()).item() for p in self.params], device=self.device)
        if self.metric=='mean': ##means often tend to be approx. 0 so hard to viz
            self.update_running[self.counter] += torch.tensor([((lr*p.grad).mean() / p.data.mean()).item() for p in self.params], device=self.device)
        self.counter += 1
        if self.counter == self.total_iters:
            self._calc_epoch_average()

    def _calc_epoch_average(self):
        ##take the mean across all iterations, finding it for each layer in params
        means = self.update_running.mean(dim=1) 
        ##store the epoch mean in the output dictionary for the appropirate layer
        for i, name in enumerate(self.names):
            self.output[name].append(means[i].item())
        
        ##reset the running tensor to zeros and the counter
        self.update_running = torch.zeros((self.total_iters, len(self.params)), device=self.device)
        self.counter = 0


def plot_UpdateRatio(ratio_dict, log10=False, exclude=['bias']):
    """
    ratio_dict -- UpdateRatioTracker.output, the layer-dictionary of update ratio lists\n
    log10 -- bool. True plots the logged update ratios instead of raw
    """
    def maybe_log(x, log=True):
        try:
            result =  math.log10(x) if log else x
        except ValueError:
            result = 0
        return result

    legends = []
    for i, (name, ratios) in enumerate(zip(ratio_dict.keys(), ratio_dict.values())):
        if any(exc in name.split('_') for exc in exclude):
            pass
        else:
            plt.plot([maybe_log(r, log10) for r in ratios])
            legends.append(name)
    plt.plot([0, len(ratios)], [-maybe_log(3, not log10), -maybe_log(3, not log10)], 'k:')
    plt.legend(legends)
    plt.figtext(0.5, 0.01, "ideal update ratio is approx. 10e-3")
    