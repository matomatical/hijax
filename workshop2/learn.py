"""
Teacher-student perceptron learning, vanilla JAX.

Notes:

* did anyone try the challenge from last week?
* new dependency `pip install plotille` for easy-ish plotting
* workshop 2 demo: stochastic gradient descent
* challenge 2: multi-layer perceptron!
"""

import time

import plotille
import tqdm
import tyro

import jax
import jax.numpy as jnp


# # # 
# Training loop


def main(
    num_steps: int = 200,
    learning_rate: float = 0.01,
    seed: int = 0,
):
    pass # TODO!


# # # 
# Perceptron architecture


# TODO!


# # # 
# Visualisation


def vis(x=None, overwrite=True, **models):
    # configure plot
    fig = plotille.Figure()
    fig.width = 40
    fig.height = 15
    fig.set_x_limits(-4, 4)
    fig.set_y_limits(-3, 3)
    
    # compute data and add to plot
    xs = jnp.linspace(-4, 4)
    for (label, w), color in zip(models.items(), ['cyan', 'magenta']):
        ys = forward_pass(w, xs)
        fig.plot(xs, ys, label=label, lc=color)
    
    # add a marker for the input batch
    if x is not None:
        fig.text([x], [0], ['x'], lc='yellow')
    
    # render to string
    figure_str = str(fig.show(legend=True))
    reset = f"\x1b[{len(figure_str.splitlines())+1}A" if overwrite else ""
    return reset + figure_str


# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)
