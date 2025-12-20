"""
Lecture 02: Hi, procedural random number generation!

Demonstration: Implement and train a classical perceptron with classical
stochastic gradient descent.

Learning objectives:

* introducing jax.random
* more practice with jax arrays, jax.grad
"""

import time
import tyro
import matthewplotlib as mp
from jaxtyping import Bool, Float, Array

import jax
import jax.numpy as jnp


# # # 
# ENTRY POINT


def main(
    num_points: int = 256,
    learning_rate: float = 0.2,
    seed: int = 42,
):
    # TODO


# # # 
# VISUALISATION CODE


def vis_data(
    xs: Float[Array, "n 2"],
    ys: Bool[Array, "n"],
) -> mp.plot:
    return mp.axes(
        mp.scatter(
            (xs[:,0], xs[:,1], mp.cyber(ys.astype(float))),
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
    w: Float[Array, "3"],
    xs: Float[Array, "n 2"],
    step: int,
) -> mp.plot:
    # compute predictions
    ys_pred = jax.nn.sigmoid(forward(w, xs))

    # compute decision boundary
    x0 = jnp.linspace(-3, 3, 80)
    y0 = -w[0]/w[1]*x0-w[2]/w[1]

    # plot
    return mp.axes(
        mp.scatter(
            (xs[:,0], xs[:,1], mp.cyber(ys_pred)),
            (xs[(step,),0], xs[(step,),1], (0,0,0)),
            (x0, y0, "white"),
            xrange=(-3,+3),
            yrange=(-3,+3),
            width=40,
            height=20,
        ),
        title=f"model predictions @ step {step+1:3d}",
        xlabel="x0",
        ylabel="x1",
    )


if __name__ == "__main__":
    tyro.cli(main)
