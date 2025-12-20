"""
Lecture 03: Hi, PyTrees!

Demonstration: Implement and train a multi-layer perceptron on XOR data with
minibatch SGD.

Learning objectives:

* more jax.numpy and jax.random
* introducing "PyTrees"
* introducing jax.tree and jax.tree_util
"""

import dataclasses
import time
import tyro
import matthewplotlib as mp
from jaxtyping import Bool, Float, Array, PRNGKeyArray
from typing import Self

import jax
import jax.numpy as jnp
import einops


# # # 
# MODEL CODE



# # # 
# TRAINING CODE

def main(
    num_points: int = 1024,
    num_steps: int = 512,
    learning_rate: float = 0.1,
    num_hidden: int = 256,
    minibatch_size: int = 64,
    seed: int = 42,
):
    # TODO


# # # 
# VISUALISATION


def vis_data(
    xs: Float[Array, "n 2"],
    ys: Bool[Array, "n"],
) -> mp.plot:
    return mp.axes(
        mp.scatter(
            (xs[:,0], xs[:,1], mp.cyber(ys)),
            xrange=(-3,+3),
            yrange=(-3,+3),
            width=40,
            height=20,
        ),
        title="ground truth labels",
        xlabel="x0",
        ylabel="x1",
    )


def vis_model(
    w: Model,
    xs: Float[Array, "n 2"],
    ys: Bool[Array, "n"],
    step: int,
) -> mp.plot:
    # compute predictions
    ys_pred = jax.nn.sigmoid(forward(w, xs)[:, 0])
    # ys_pred = jax.nn.sigmoid(w.forward(xs)[:, 0])

    # plot
    return mp.axes(
        mp.dstack2(
            mp.function2(
                F=lambda xs: jax.nn.sigmoid(forward(w, xs)[:,0]),
                # F=lambda xs: jax.nn.sigmoid(w.forward(xs)[:,0]),
                xrange=(-3,3),
                yrange=(-3,3),
                width=40,
                height=20,
                zrange=(0., 1.),
                colormap=lambda z: 0.5 * (mp.cyber(z) / 255),
                endpoints=True,
            ),
            mp.scatter(
                (xs[:,0], xs[:,1], mp.cyber(ys)),
                xrange=(-3,+3),
                yrange=(-3,+3),
                width=40,
                height=20,
            ),
        ),
        title=f"model predictions @ step {step+1:3d}",
        xlabel="x0",
        ylabel="x1",
    )


if __name__ == "__main__":
    tyro.cli(main)
