"""
Lecture 1: Hi, automatic differentiation!

Demonstration: Train a teacher--student linear regression model with gradient
descent.

Learning objectives:

* more jax.numpy
* introducing functional model API
* introducing jax.grad
"""

import time
import tyro
import matthewplotlib as mp
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp


def main(
    num_steps: int = 400,
    learning_rate: float = 0.01,
):
    # TODO

def vis(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
    step: int,
    loss: float,
) -> mp.plot:
    x = jnp.linspace(-4, 4, 80)
    return mp.axes(
        mp.scatter(
            mp.xaxis(-4, 4, 80),
            mp.yaxis(-4, 4, 80),
            # TODO
            height=20,
            width=40,
            xrange=(-4,4),
            yrange=(-4,4),
        ),
        title=f"step {step:03d} | loss {loss:6.3f}",
        ylabel="y",
        xlabel="x",
    )

if __name__ == "__main__":
    tyro.cli(main)
