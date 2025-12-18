"""
Lecture 04: Hi, automatic vectorisation!

Demonstration: Implement and train a CNN on MNIST with minibatch SGD.

Learning objectives:

* more practice with PyTrees
* introducing jax.vmap
"""

import functools
import dataclasses
import time
import tyro
import matthewplotlib as mp

from jaxtyping import Int, Float, Array, PRNGKeyArray
from typing import Self

import numpy as np
import jax
import jax.numpy as jnp
import einops


# # # 
# Training loop


def main(
    learning_rate: float = 0.05,
    batch_size: int = 512,
    num_steps: int = 256,
    steps_per_visualisation: int = 4,
    seed: int = 42,
):
    key = jax.random.key(seed)


    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = SimpLeNet.init(key=key_model)

    print(jax.tree.map(lambda l: jnp.array(l).shape, model))


    print("loading and preprocessing data...")
    with jnp.load('mnist.npz') as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])
    x_train = 1.275 * x_train / 255 - 0.1
    x_test = 1.275 * x_test / 255 - 0.1
    
    probs = model.forward(x_train[0])
    print(vis_digits(
        digits=x_train[:6],
        true_labels=y_train[:6],
        model=model,
    ))

    
    print("begin training...")
    losses = []
    accuracies = []
    plots = []
    for step in range(num_steps):
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

        # compute update, update optimiser and model
        model = jax.tree.map(
            lambda leaf_w, leaf_g: leaf_w - learning_rate * leaf_g,
            model,
            grads,
        )

        # track metrics
        losses.append((step, loss))
        test_acc = batch_accuracy(model, x_test[:1000], y_test[:1000])
        accuracies.append((step, test_acc))

        # visualisation!
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            digit_plot = vis_digits(
                digits=x_test[(1,6,7,8),],
                true_labels=y_test[(1,6,7,8),],
                model=model,
            )
            metrics_plot = vis_metrics(
                losses=losses,
                accuracies=accuracies,
                total_num_steps=num_steps,
            )
            plot = digit_plot / metrics_plot
            if step == 0:
                print(plot)
            else:
                print(f"\x1b[{plot.height}A{plot}")
            plots.append(plot)


    # mp.save_animation(
    #     plots,
    #     "../gallery/lecture04.gif",
    #     bgcolor="black",
    #     fps=10,
    # )


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


def scaled_tanh(x):
    return 1.7159 * jnp.tanh(0.6667 * x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Subsample2x2:
    weights: Float[Array, "c"]
    biases: Float[Array, "c"]

    @staticmethod
    def init(num_channels: int) -> Self:
        return Subsample2x2(
            weights=jnp.ones((num_channels, 1, 1)) / 4,
            biases=jnp.zeros((num_channels, 1, 1)),
        )

    def forward(w, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        sums = einops.reduce(x, 'c (h 2) (w 2) -> c h w', 'sum')
        return w.weights * sums + w.biases


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["kernel"],
    meta_fields=["padding"],
)
@dataclasses.dataclass
class SimpleConv2d:
    kernel: Float[Array, "k k c_in c_out"]
    padding: int

    @staticmethod
    def init(
        key: PRNGKeyArray,
        kernel_size: int,
        num_channels_in: int,
        num_channels_out: int,
        padding: int,
    ) -> Self:
        num_inputs = num_channels_in * kernel_size**2
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return SimpleConv2d(
            kernel=jax.random.uniform(
                key=key,
                shape=(
                    kernel_size,
                    kernel_size,
                    num_channels_in,
                    num_channels_out,
                ),
                minval=-bound,
                maxval=+bound,
            ),
            padding=[(padding, padding), (padding, padding)],
        )

    def forward(
        w: Self,
        x: Float[Array, "c_in h w"],
    ) -> Float[Array, "c_out h_ w_"]:
        x_1chw = einops.rearrange(x, 'c h w -> 1 c h w')
        y_1chw = jax.lax.conv_general_dilated(
            lhs=x_1chw,
            rhs=w.kernel,
            window_strides=(1,1),
            padding=w.padding,
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        y_chw = y_1chw[0]
        return y_chw


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class SimpLeNet:
    C1: SimpleConv2d
    S2: Subsample2x2
    C3: SimpleConv2d
    S4: Subsample2x2
    C5: SimpleConv2d
    F6: AffineTransform
    Out: AffineTransform

    @staticmethod
    def init(key: PRNGKeyArray) -> Self:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return SimpLeNet(
            C1=SimpleConv2d.init(
                key=k1,
                kernel_size=5,
                num_channels_in=1,
                num_channels_out=6,
                padding=2,
            ),
            S2=Subsample2x2.init(num_channels=6),
            C3=SimpleConv2d.init(
                key=k2,
                kernel_size=5,
                num_channels_in=6,
                num_channels_out=16,
                padding=0,
            ),
            S4=Subsample2x2.init(num_channels=16),
            C5=SimpleConv2d.init(
                key=k3,
                kernel_size=5,
                num_channels_in=16,
                num_channels_out=120,
                padding=0,
            ),
            F6=AffineTransform.init(
                key=k4,
                num_inputs=120,
                num_outputs=84,
            ),
            Out=AffineTransform.init(
                key=k5,
                num_inputs=84,
                num_outputs=10,
            ),
        )

    def forward(
        self: Self,
        image: Float[Array, "28 28"],
    ) -> Float[Array, "10"]:
        # Input:       1x28x28
        x = einops.rearrange(image, 'h w -> 1 h w')
        # C1:       -> 6x28x28
        x = scaled_tanh(self.C1.forward(x))
        # S2:       -> 6x14x14
        x = scaled_tanh(self.S2.forward(x))
        # C3*:      -> 16x10x10 (note: fully connected channels)
        x = scaled_tanh(self.C3.forward(x))
        # S4:       -> 16x5x5
        x = scaled_tanh(self.S4.forward(x))
        # C5:       -> 120x1x1 (note: equiv. dense 400->120 at this size)
        x = scaled_tanh(self.C5.forward(x))
        # (flatten) -> 120
        x = jnp.ravel(x)
        # F6:       -> 84
        x = scaled_tanh(self.F6.forward(x))
        # Output*:  -> 10 (note: learned map, no hand-made RBF code)
        x = self.Out.forward(x)
        return jax.nn.softmax(x)

    def forward_batch(
        self,
        images: Float[Array, "batch_size 28 28"],
    ) -> Float[Array, "batch_size 10"]:
        forward = SimpLeNet.forward
        vforward = jax.vmap(
            forward,
            in_axes=(None, 0),
            out_axes=0,
        )
        return vforward(self, images)


# # # 
# Metrics


def cross_entropy(
    model: SimpLeNet,
    x: Float[Array, "h w"],
    y: int,
) -> float:
    # Cross entropy formula: Hx(q, p) = - Sum_i p(i) log q(i)
    return -jnp.log(model.forward(x)[y])


def batch_cross_entropy(
    model: SimpLeNet,
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


def batch_accuracy(
    model: SimpLeNet,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    pred_prob_all_classes = model.forward_batch(x_batch)
    highest_prob_class = pred_prob_all_classes.argmax(axis=-1)
    return jnp.mean(y_batch == highest_prob_class)


# # # 
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    true_labels: Int[Array, "n"],
    model: SimpLeNet,
) -> mp.plot:
    # shrink and normalise images
    ddigits = einops.reduce(
        (digits + 0.1) / 1.275,
        'b (h 2) (w 2) -> b h w',
        'mean',
    )
    width = ddigits.shape[-1]

    # classify digits and mark correct or incorrect
    pred_probs = model.forward_batch(digits)
    pred_labels = pred_probs.argmax(axis=-1)
    corrects = (true_labels == pred_labels)
    cmaps = [mp.cyans if correct else mp.magentas for correct in corrects]

    # build the visualisation
    array = mp.wrap(*[
        mp.blank(width=3)
        + mp.border(
            mp.columns(
                probs,
                height=6,
                vrange=1,
                column_width=1,
                column_spacing=1,
                colors=[mp.cyber(i==label) for i in range(10)],
            ) / mp.text(" ".join(str(d) for d in range(10)))
            + mp.blank(width=2)
            + mp.image(ddigit, colormap=cmap),
            title="p( digit | image )──────image────",
        )
        for ddigit, label, probs, cmap in zip(ddigits, true_labels, pred_probs, cmaps)
    ], cols=2)
    return array


def vis_metrics(
    losses: list[tuple[int, float]],
    accuracies: list[tuple[int, float]],
    total_num_steps: int,
) -> mp.plot:
    losses = np.asarray(losses)
    accuracies = np.asarray(accuracies)
    loss_plot = mp.axes(
        mp.scatter(
            (losses, 'magenta'),
            xrange=(0, total_num_steps-1),
            yrange=(0, max(l for s, l in losses)),
            width=35,
            height=11,
        ),
        title=f"cross entropy {losses[-1][1]:.3f}",
        xlabel="train steps",
    )
    acc_plot = mp.axes(
        mp.scatter(
            (accuracies, 'cyan'),
            xrange=(0, total_num_steps-1),
            yrange=(0, 1),
            width=35,
            height=11,
        ),
        title=f"test accuracy {accuracies[-1][1]:.2%}",
        xlabel="train steps",
    )
    return loss_plot + acc_plot


# # # 
# Entry point

if __name__ == "__main__":
    tyro.cli(main)
