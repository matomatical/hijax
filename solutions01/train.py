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
    # define teacher
    w_teacher = jnp.array([0.5, -1.0])

    # define student
    w_student = jnp.array([-1.0, 3.0])
    
    print(vis(w_student, w_teacher, step=0, loss=jnp.inf))

    # training loop
    for step in range(1, num_steps+1):
        # l = loss(w_student, w_teacher)
        loss_and_grad_fn = jax.value_and_grad(loss)
        l, g = loss_and_grad_fn(w_student, w_teacher)

        w_student = w_student - learning_rate * g

        plot = vis(w_student, w_teacher, step, l)
        print(f"\x1b[{plot.height}A{plot}")
        time.sleep(0.02)


def loss(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
) -> float:
    x = jnp.linspace(-4, 4, 80)
    y_student = forward(w_student, x)
    y_teacher = forward(w_teacher, x)
    errors = y_teacher - y_student
    squared_errors = errors ** 2
    return jnp.mean(squared_errors)


def forward(
    w: Float[Array, "2"],
    x: Float[Array, "batch_size"],
) -> Float[Array, "batch_size"]:
    a, b = w
    return a * x + b


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
            (x, forward(w_teacher, x), 'cyan'),
            (x, forward(w_student, x), 'magenta'),
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
