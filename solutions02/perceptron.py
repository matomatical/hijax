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
    # initialise the PRNG
    key = jax.random.key(seed=seed)

    # initialise a data set
    key_pos, key = jax.random.split(key)
    xs_pos = jax.random.multivariate_normal(
        key=key_pos,
        mean=jnp.ones(2),
        cov=0.25 * jnp.eye(2),
        shape=(num_points//2,),
    )
    ys_pos = jnp.ones(num_points//2, dtype=bool)

    key_neg, key = jax.random.split(key)
    xs_neg = jax.random.multivariate_normal(
        key=key_neg,
        mean=-jnp.ones(2),
        cov=0.25 * jnp.eye(2),
        shape=(num_points//2,),
    )
    ys_neg = jnp.zeros(num_points//2, dtype=bool)

    xs = jnp.concatenate([xs_pos, xs_neg], axis=0)
    ys = jnp.concatenate([ys_pos, ys_neg], axis=0)

    print(vis_data(xs, ys))

    # shuffle data
    key_shuffle, key = jax.random.split(key)
    pi = jax.random.permutation(
        key=key_shuffle,
        x=ys.size,
    )
    xs = xs[pi]
    ys = ys[pi]

    # define perceptron model
    key_model_init, key = jax.random.split(key)
    w = jax.random.normal(
        key=key_model_init,
        shape=(3,),
    )

    # training loop
    print(vis_model(w, xs, step=0))
    for t, (x, y) in enumerate(zip(xs, ys)):
        l, g = jax.value_and_grad(loss)(w, x, y)
        w = w - learning_rate * g

        plot = vis_model(w, xs, step=t)
        print(f"{-plot}{plot}")
        time.sleep(0.02)


def loss(
    w: Float[Array, "3"],
    x: Float[Array, "2"],
    y: Bool[Array, ""],
) -> Float[Array, ""]:
    logit = forward(w, x)
    cross_entropy = jnp.logaddexp(0, logit) - y * logit
    return cross_entropy


def forward(
    w: Float[Array, "3"],
    x: Float[Array, "2"],
) -> float:
    a = w[:2]
    b = w[2]
    return jnp.dot(x, a) + b


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
