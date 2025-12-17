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
    # initialise teacher
    w_teacher = jnp.array([.5, -1.])

    # initialise student
    w_student = jnp.array([-1., 3.])

    # initialise input data
    x = jnp.linspace(-4, 4, 80)

    # training loop
    plot = vis(w_student, w_teacher, x, 0)
    plots = [plot]
    print(plot)
    for t in range(num_steps):
        l = loss(w_student, w_teacher, x)
        g_student = jax.grad(loss)(w_student, w_teacher, x)
        w_student = w_student - learning_rate * g_student
        plot = vis(w_student, w_teacher, x, t+1)
        print(f"{-plot}{plot}")
        plots.append(plot)
        time.sleep(0.02)

    mp.save_animation(
        plots,
        "../gallery/lecture01.gif",
        bgcolor="black",
        fps=50,
    )


def forward(
    w: Float[Array, "2"],
    x: Float[Array, "batch_size"],
) -> Float[Array, "batch_size"]:
    a, b = w
    return a * x + b


def loss(w_student, w_teacher, x):
    diff = forward(w_student, x) - forward(w_teacher, x)
    return jnp.mean(diff**2)


def vis(
    w_student: Float[Array, "2"],
    w_teacher: Float[Array, "2"],
    x: Float[Array, "batch_size"],
    step: int,
) -> mp.plot:
    return mp.axes(
        mp.scatter(
            mp.xaxis(-4, 4, 80),
            mp.yaxis(-4, 4, 80),
            (x, forward(w_student, x), 'magenta'),
            (x, forward(w_teacher, x), 'cyan'),
            height=20,
            width=40,
            xrange=(-4,4),
            yrange=(-4,4),
        ),
        title=f"step {step}",
        ylabel="y",
        xlabel="x",
    )

if __name__ == "__main__":
    tyro.cli(main)
