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

# type Model = tuple[
#     Float[Array, "2 hidden"],
#     Float[Array, "hidden"],
#     Float[Array, "hidden 1"],
#     Float[Array, "1"],
# ]


# def forward(
#     w: Model,
#     xs: Float[Array, "batch_size 2"],
# ) -> Float[Array, "batch_size 1"]:
#     W1, b1, W2, b2 = w
#     # layer 1
#     xs = xs @ W1 + b1
#     # activation
#     xs = jax.nn.relu(xs)
#     # layer 2
#     logits = xs @ W2 + b2
#     return logits


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Model:
    W1: Float[Array, "2 hidden"]
    b1: Float[Array, "hidden"]
    W2: Float[Array, "hidden 1"]
    b2: Float[Array, "1"]
    
    @staticmethod
    def init(
        key: PRNGKeyArray,
        num_hidden: int,
    ) -> Self:
        key_W1, key = jax.random.split(key)
        W1 = jax.random.normal(
            key=key_W1,
            shape=(2, num_hidden),
        )
        b1 = jnp.zeros(num_hidden)
        key_W2, key = jax.random.split(key)
        W2 = jax.random.normal(
            key=key_W2,
            shape=(num_hidden, 1),
        )
        b2 = jnp.zeros(1)
        return Model(W1=W1, b1=b1, W2=W2, b2=b2)

    def forward(
        self: Self,
        xs: Float[Array, "batch_size 2"],
    ) -> Float[Array, "batch_size 1"]:
        # layer 1
        xs = xs @ self.W1 + self.b1
        # activation
        xs = jax.nn.relu(xs)
        # layer 2
        logits = xs @ self.W2 + self.b2
        return logits


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
    key = jax.random.key(seed=seed)

    # initialise some training data
    key_data, key = jax.random.split(key)
    xs = jax.random.multivariate_normal(
        key=key_data,
        mean=jnp.zeros(2),
        cov=jnp.eye(2),
        shape=num_points,
    )
    cs = einops.repeat(
        jnp.arange(4),
        'n -> (n k)',
        k=num_points//4,
    )
    xs = 0.5 * xs + jnp.array([
        (-1,-1),
        (+1,+1),
        (-1,+1),
        (+1,-1),
    ])[cs]
    ys = cs // 2

    print(vis_data(xs, ys))

    # initialise model
    key_model_init, key = jax.random.split(key)
    # key_W1, key_model_init = jax.random.split(key_model_init)
    # W1 = jax.random.normal(
    #     key=key_W1,
    #     shape=(2, num_hidden),
    # )
    # b1 = jnp.zeros(num_hidden)
    # key_W2, key_model_init = jax.random.split(key_model_init)
    # W2 = jax.random.normal(
    #     key=key_W2,
    #     shape=(num_hidden, 1),
    # )
    # b2 = jnp.zeros(1)
    # w = (W1, b1, W2, b2)
    w = Model.init(key=key_model_init, num_hidden=num_hidden)

    print(vis_model(w, xs, ys, -1))

    key_train, key = jax.random.split(key)
    for t in range(num_steps):
        # sample a minibatch
        key_minibatch, key_train = jax.random.split(key_train)
        minibatch_idx = jax.random.choice(
            key=key_minibatch,
            a=num_points,
            shape=(minibatch_size,),
            replace=False,
        )
        xs_batch = xs[minibatch_idx]
        ys_batch = ys[minibatch_idx]

        # do a gradient step
        l, g = jax.value_and_grad(loss)(w, xs_batch, ys_batch)
        w = jax.tree.map(
            lambda model_leaf, grad_leaf:
                model_leaf - learning_rate * grad_leaf,
            w,
            g,
        )

        # visualise
        plot = vis_model(w, xs, ys, t)
        print(f"{-plot}{plot}")
        time.sleep(0.02)




def loss(
    w: Model,
    xs: Float[Array, "batch_size 2"],
    ys: Bool[Array, "batch_size"],
) -> float:
    # logits = forward(w, xs)[:, 0]
    logits = w.forward(xs)[:, 0]
    cross_entropies = jnp.logaddexp(0, logits) - ys * logits
    return jnp.mean(cross_entropies)


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
    # ys_pred = jax.nn.sigmoid(forward(w, xs)[:, 0])
    ys_pred = jax.nn.sigmoid(w.forward(xs)[:, 0])

    # plot
    return mp.axes(
        mp.dstack2(
            mp.function2(
                # F=lambda xs: jax.nn.sigmoid(forward(w, xs)[:,0]),
                F=lambda xs: jax.nn.sigmoid(w.forward(xs)[:,0]),
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
