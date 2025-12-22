"""
Lecture 07: Hi, loop acceleration!

Demonstration: Implement a deep dense residual network for MNIST.

Learning objectives:

* further exploration of jax.jit (jit dojo):
  * loops are unrolled at trace time
* jax.lax.for_i and jax.lax.scan
"""

import functools
import dataclasses
import time
import tyro
# import tqdm
import jax_tqdm
import matthewplotlib as mp

from jaxtyping import Int, Float, Array, PRNGKeyArray, PyTree
from typing import Self

import numpy as np
import jax
import jax.numpy as jnp
import einops
import pcax


# # # 
# Training loop


def main(
    learning_rate: float = 0.001,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    batch_size: int = 512,
    num_steps: int = 256,
    seed: int = 42,
):
    key = jax.random.key(seed)


    print("loading and preprocessing data...")
    with jnp.load('../data/mnist.npz') as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])
    x_train = 1.275 * x_train / 255 - 0.1
    x_test = 1.275 * x_test / 255 - 0.1

    
    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = DenseResNet.init(key=key_model)


    print("initialising optimiser...")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
        beta1=adam_beta1,
        beta2=adam_beta2,
    )
    

    print("begin training...")
    @jax_tqdm.loop_tqdm(num_steps)
    def train_step(i: int, val: tuple) -> tuple:
        key, opt_state, model = val

        # sample a batch
        key_batch, key = jax.random.split(key)
        batch = jax.random.choice(
            key=key_batch,
            a=y_train.size,
            shape=(batch_size,),
            replace=False,
        )
        x_batch = x_train[batch]
        y_batch = y_train[batch]

        # compute the batch loss and grad
        grads = jax.grad(batch_cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # compute update, update optimiser, update model
        delta, opt_state = opt_state.update(grads)
        model = jax.tree.map(jnp.add, model, delta)

        return key, opt_state, model

    key, opt_state, model = jax.lax.fori_loop(
        lower=0,
        upper=num_steps,
        body_fun=train_step,
        init_val=(key, opt_state, model),
    )

    print("PCA of trajectories...")
    # capture activations
    X = x_test[:1000]
    Y = y_test[:1000]
    Z: Float[Array, "1000 201 128"]
    _Y, Z = jax.vmap(model.forward_activations)(X)

    # pca
    Z_combined = einops.rearrange(
        Z,
        'batch depth width -> (batch depth) width',
    )
    pca = pcax.fit(Z_combined, n_components=3)
    xyz_bd = pcax.transform(pca, Z)

    # plotting
    xyz_bd = np.array(xyz_bd)
    xyz_bd = xyz_bd / np.abs(xyz_bd).max()
    colours = mp.cyber(Y / 9)

    for t in range(500):
        # points to plot
        layer = min(200, t)
        xyz_b = xyz_bd[:, layer]

        # sweep camera
        angle = t / 500 * 4 * np.pi
        p = np.array([
            1.5 * np.sin(angle),
            1.5,
            1.5 * np.cos(angle),
        ])
        
        # construct plot
        plot = mp.border(
            mp.scatter3(
                (mp.xaxis(0, 0.2), 'red'),
                (mp.yaxis(0, 0.2), 'green'),
                (mp.zaxis(0, 0.2), 'blue'),
                (xyz_b, colours),
                camera_position=p,
                vertical_fov_degrees=55,
                height=36,
                width=78,
            ),
            title=f"PCA of activations at layer {layer}"
        )
        if t == 0:
            print(plot)
        else:
            print(f"{-plot}{plot}")
        time.sleep(0.02)


# # # 
# Architecture


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AffineTransform:
    weights: Float[Array, "n_in n_out"]
    biases: Float[Array, "n_out"]

    @staticmethod
    def init(key: PRNGKeyArray, num_inputs: int, num_outputs: int) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return AffineTransform(
            weights=jax.random.uniform(
                key=key,
                shape=(num_inputs, num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
            biases=jnp.zeros(num_outputs),
        )

    def forward(
        w: Self,
        x: Float[Array, "n_in"],
    ) -> Float[Array, "n_out"]:
        return x @ w.weights + w.biases


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DenseResNet:
    embedding: AffineTransform      # 784 -> 128
    layers: AffineTransform         # AffineTransform[depth] (128 -> 128)
    unembedding: AffineTransform    # 128 -> 10

    @jax.jit
    def forward(
        self: Self,
        image: Float[Array, "28 28"],
    ) -> Float[Array, "10"]:
        # embed
        x = jnp.ravel(image)
        x = self.embedding.forward(x)
        x = jnp.tanh(x)

        # layers
        # for layer in self.layers:
        #     r = layer.forward(x)
        #     r = jnp.tanh(r)
        #     x = x + r
        def step(x, layer):
            r = layer.forward(x)
            r = jnp.tanh(r)
            x = x + r
            return x, None
        x, _ = jax.lax.scan(
            step,
            x,
            self.layers,
        )

        # unembed
        logits = self.unembedding.forward(x)
        probs = jax.nn.softmax(logits)
        return probs
    
    @jax.jit
    def forward_activations(
        self: Self,
        image: Float[Array, "28 28"],
    ) -> tuple[
        Float[Array, "10"],
        Float[Array, "201 128"],
    ]:
        # embed
        x = jnp.ravel(image)
        x = self.embedding.forward(x)
        x = jnp.tanh(x)
        embedding = x

        def step(x, layer):
            r = layer.forward(x)
            r = jnp.tanh(r)
            x = x + r
            return x, x
        x, activations = jax.lax.scan(
            step,
            x,
            self.layers,
        )

        # unembed
        logits = self.unembedding.forward(x)
        probs = jax.nn.softmax(logits)

        # concat activations
        all_activations = jnp.concatenate(
            [embedding[None], activations],
            axis=0,
        )
        return probs, all_activations

    @staticmethod
    @jax.jit
    def init(
        key: PRNGKeyArray,
    ) -> Self:
        k1, k2, k3 = jax.random.split(key, 3)
        return DenseResNet(
            embedding=AffineTransform.init(
                key=k1,
                num_inputs=784,
                num_outputs=128,
            ),
            # layers=[
            #     AffineTransform.init(
            #         key=k,
            #         num_inputs=128,
            #         num_outputs=128,
            #     )
            #     for k in jax.random.split(k2, 200)
            # ],
            layers=jax.vmap(
                AffineTransform.init,
                in_axes=(0, None, None),
                out_axes=AffineTransform(weights=0, biases=0), # optional
            )(
                jax.random.split(k2, 200),
                128,
                128,
            ),
            unembedding=AffineTransform.init(
                key=k3,
                num_inputs=128,
                num_outputs=10,
            ),
        )


# # # 
# Optimiser


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Adam:
    moment1: PyTree["model"]
    moment2: PyTree["model"]
    alpha: float
    beta1: float
    beta2: float
    time: int

    @staticmethod
    @jax.jit
    def init(
        model: PyTree["model"],
        alpha: float,
        beta1: float,
        beta2: float,
    ) -> Self:
        return Adam(
            moment1=jax.tree.map(jnp.zeros_like, model),
            moment2=jax.tree.map(jnp.zeros_like, model),
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
            time=0,
        )

    @jax.jit
    def update(
        self: Self,
        grads: PyTree["model"],
    ) -> tuple[
        PyTree["model"],
        Self,
    ]:
        # update optimiser state
        t = self.time + 1
        moment1 = jax.tree.map(
            lambda m1, g: self.beta1 * m1 + (1-self.beta1) * g,
            self.moment1,
            grads,
        )
        moment2 = jax.tree.map(
            lambda m2, g: self.beta2 * m2 + (1-self.beta2) * g**2,
            self.moment2,
            grads,
        )
        new_state = dataclasses.replace(
            self,
            moment1=moment1,
            moment2=moment2,
            time=t,
        )

        # compute model update from optimiser state
        moment1_unbiased = jax.tree.map(
            lambda m1: m1 / (1-self.beta1**t),
            new_state.moment1,
        )
        moment2_unbiased = jax.tree.map(
            lambda m2: m2 / (1-self.beta2**t),
            new_state.moment2,
        )
        update = jax.tree.map(
            lambda m1, m2: - self.alpha * m1 / (jnp.sqrt(m2) + 1e-8),
            moment1_unbiased,
            moment2_unbiased,
        )
        return update, new_state


# # # 
# Metrics


@jax.jit
def cross_entropy(
    model: DenseResNet,
    x: Float[Array, "h w"],
    y: int,
) -> float:
    # Cross entropy formula: Hx(q, p) = - Sum_i p(i) log q(i)
    return -jnp.log(model.forward(x)[y])


@jax.jit
def batch_cross_entropy(
    model: DenseResNet,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    """
    Average cross entropy from across the batch.
    """
    vmapped_cross_entropy_fn = jax.vmap(
        cross_entropy,
        in_axes=(None,0,0),
    )
    all_cross_entropies = vmapped_cross_entropy_fn(
        model,
        x_batch,
        y_batch,
    )
    avg_cross_entropy = all_cross_entropies.mean()
    return avg_cross_entropy


# # # 
# Entry point

if __name__ == "__main__":
    tyro.cli(main)
