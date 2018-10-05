"""
Module containing small functions used to configure matplotlib figures.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def configure_subplots(fig, ax, axis, xticks, yticks, xlabel, ylabel, 
                       hideleft=False, xticklabels=None, xticklabelrot=0):
    set_shared_labels(fig, xlabel, ylabel)
    simpleaxis(ax, left=hideleft)
    set_xticks(ax, xticks)
    set_yticks(ax, yticks)
    set_axis(ax, axis)
    if xticklabels is not None:
        set_xticklabels(ax, xticklabels, rot=xticklabelrot)

def simpleax(ax, left=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def simpleaxis(axes, left=False):
    for ax in axes.ravel():
           simpleax(ax, left=left)

def noaxis(axes):
    for ax in axes.ravel():
        ax.axis('off')

def set_xticks(axes, xticks):
    for ax in axes.ravel():
        ax.set_xticks(xticks)

def set_xticklabels(axes, labels, rot=0):
    for ax in axes.ravel():
        ax.set_xticklabels(labels, rotation=rot)

def set_yticks(axes, yticks):
    for ax in axes.ravel():
        ax.set_yticks(yticks)

def set_axis(axes, axis):
    for ax in axes.ravel():
        ax.axis(axis)

def set_shared_labels(fig, xlabel, ylabel):   
    fig.text(0.51, -0.04, xlabel, ha='center')
    fig.text(0.07, 0.51, ylabel, va='center', rotation='vertical')
