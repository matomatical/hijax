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
import tqdm
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


    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = DenseResNet.init(key=key_model)


    print("loading and preprocessing data...")
    with jnp.load('mnist.npz') as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])
    x_train = 1.275 * x_train / 255 - 0.1
    x_test = 1.275 * x_test / 255 - 0.1


    print("initialising optimiser...")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
        beta1=adam_beta1,
        beta2=adam_beta2,
    )
    

    print("begin training...")
    losses = []
    accuracies = []
    plots = []
    # TODO: scan this loop
    for step in tqdm.trange(num_steps):
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
        loss, grads = jax.value_and_grad(batch_cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # compute update, update optimiser, update model
        delta, opt_state = opt_state.update(grads)
        model = jax.tree.map(jnp.add, model, delta)


    print("begin dynamics analysis...")
    
    # extract activations
    X = x_test[:1000]
    Y = y_test[:1000]
    Y_, Z = model.batch_forward_activations(X)

    # PCA
    Z_all = einops.rearrange(
        Z,
        'batch layers residual -> (batch layers) residual',
    )
    pca = pcax.fit(Z_all, n_components=3)
    xyz = pcax.transform(pca, Z)
    
    # prepare to plot
    xyz_np = np.array(xyz)
    xyz_std = xyz_np / np.abs(xyz_np).max()
    
    # plot evolving representation cloud
    plot = None
    plots = []
    for t in range(500):
        l = min(t, 200)
        # sweep camera
        angle = t / 250 * 4 * np.pi
        p = np.array([1.5 * np.sin(angle), 1.5, 1.5 * np.cos(angle)])

        # determine points and colours
        series = xyz_std[:,l,:]
        c = mp.cyber(Y/10)
        highlighted_class = (t // 25) % 10
        c[Y == highlighted_class] = 255

        # plot
        if plot:
            print(-plot, end="")
        plot = mp.border(
            (mp.text("layers: ") + mp.progress(l/200, width=70))
            / mp.scatter3(
                (mp.xaxis(0,.2), "red"),
                (mp.yaxis(0,.2), "green"),
                (mp.zaxis(0,.2), "blue"),
                (series, c),
                camera_position=p,
                vertical_fov_degrees=55,
                height=36,
                width=78,
            )
            / mp.text(f"highlighted class: {highlighted_class}"),
            title=f"PCA representations by class",
        )
        print(plot)
        plots.append(plot)
        time.sleep(0.02)


    mp.save_animation(
        plots,
        "../gallery/lecture07.gif",
        bgcolor="black",
        fps=50,
    )


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
    embedding: AffineTransform
    layers: list[AffineTransform]
    unembedding: AffineTransform

    @staticmethod
    @jax.jit
    def init(key: PRNGKeyArray) -> Self:
        k1, k2, k3 = jax.random.split(key, 3)
        return DenseResNet(
            embedding=AffineTransform.init(
                key=k1,
                num_inputs=784,
                num_outputs=128,
            ),
            # TODO: vmap
            layers=[
                AffineTransform.init(
                    key=k,
                    num_inputs=128,
                    num_outputs=128,
                ) for k in jax.random.split(k2, 200)
            ],
            unembedding=AffineTransform.init(
                key=k3,
                num_inputs=128,
                num_outputs=10,
            ),
        )

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
        # TODO: Scan
        for layer in self.layers:
            r = layer.forward(x)
            r = jnp.tanh(r)
            x = x + r
        
        # unembed
        x = self.unembedding.forward(x)
        probs = jax.nn.softmax(x)
        return probs
    
    @jax.jit
    def batch_forward(
        self,
        images: Float[Array, "batch_size 28 28"],
    ) -> Float[Array, "batch_size 10"]:
        return jax.vmap(self.forward)(images)
    
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

        # layers
        # TODO: scan
        activations = [x]
        for layer in self.layers:
            r = layer.forward(x)
            r = jnp.tanh(r)
            x = x + r
            activations.append(x)
        
        # unembed
        x = self.unembedding.forward(x)
        probs = jax.nn.softmax(x)
        return probs, jnp.stack(activations)

    @jax.jit
    def batch_forward_activations(
        self: Self,
        images: Float[Array, "batch_size 28 28"],
    ) -> tuple[
        Float[Array, "batch_size 10"],
        Float[Array, "batch_size 201 128"],
    ]:
        return jax.vmap(self.forward_activations)(images)


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


@jax.jit
def batch_accuracy(
    model: DenseResNet,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    pred_prob_all_classes = model.batch_forward(x_batch)
    highest_prob_class = pred_prob_all_classes.argmax(axis=-1)
    return jnp.mean(y_batch == highest_prob_class)


# # # 
# Entry point

if __name__ == "__main__":
    tyro.cli(main)
