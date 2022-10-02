import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch


def plot_grad_flow_basic(raw_parameters : list, param_names : list=None, y_ax_max="max_average"):
    '''
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Provide the list of raw model parameters. Optionally provide list of names matching param order.
    y_ax_max is the y axis limit and is either "max_average" or "mean_average"
    Adapted from source: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    '''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in enumerate(raw_parameters):
        if(p.requires_grad) & (p.grad is not None): #and ("bias" not in n):
            layers.append(str(n))
            ave_grads.append(p.grad.abs().mean().item())
            max_grads.append(p.grad.abs().max().item())

    xticks = range(0,len(ave_grads), 1)
    y_min = min(ave_grads)
    y_max = max(ave_grads)
    if y_ax_max == "mean_average":
        y_max = (sum(ave_grads)/len(ave_grads))
    if param_names is not None:
        for x_ix in xticks:
            plt.text(x_ix, y_max/3, s=param_names[x_ix], rotation=90)

    plt.bar(range(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(range(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(xticks, layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = y_min, top=y_max) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])