"""
Lecture 04: Hi, automatic vectorisation!

Demonstration: Implement and train a CNN on MNIST with minibatch SGD.

Learning objectives:

* more practice with PyTrees
* introducing jax.vmap
"""

import functools
import dataclasses
import tyro
import matthewplotlib as mp

from jaxtyping import Int, Float, Array, PRNGKeyArray, PyTree
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
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    seed: int = 42,
    steps_per_visualisation: int = 4,
):
    key = jax.random.key(seed=seed)
    
    print("load the training data...")
    with jnp.load("../data/mnist.npz") as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])

    print(x_train.shape, x_train.dtype)
    print(x_test.shape, x_test.dtype)
    print(y_train.shape, y_train.dtype)
    print(y_test.shape, y_test.dtype)

    print(mp.image(x_train[0]))
    print(y_train[0])

    x_train = 1.275 * x_train / 255 - 0.1
    x_test = 1.275 * x_test / 255 - 0.1

    
    print("initialising the model...")
    key_model_init, key = jax.random.split(key)
    model = SimpLeNet.init(
        key=key_model_init,
    )
    
    print(sum([leaf.size for leaf in jax.tree.leaves(model)]))

    print(model.forward(x_train[0]))

    
    print("initialise the optimiser")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
        beta1=adam_beta1,
        beta2=adam_beta2,
    )
        

    print("defining training step...")
    @jax.jit
    def train_step(key, opt_state, model):
        # sample a minibatch
        key_batch, key = jax.random.split(key)
        batch_idx = jax.random.choice(
            key=key_batch,
            a=y_train.size,
            shape=(batch_size,),
            replace=False,
        )
        x_batch = x_train[batch_idx]
        y_batch = y_train[batch_idx]

        # compute batch loss and grads
        loss, grads = jax.value_and_grad(batch_cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # update optimiser, get model update and apply to model
        update, opt_state = opt_state.update(grads)
        model = jax.tree.map(
            jnp.add,
            model,
            update,
        )
        return key, opt_state, model, loss


    print("training...")
    losses = []
    accuracies = []
    for step in range(num_steps):
        # doing the jittable computation of the train step
        key, opt_state, model, loss = train_step(key, opt_state, model)

        # tracks metrics
        losses.append((step, loss))

        # visualisation
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            # evaluate
            test_acc = batch_accuracy(model, x_test[:1000], y_test[:1000])
            accuracies.append((step, test_acc))

            # visualise
            digit_plot = vis_digits(
                digits=x_test[(1,6,7,8),],
                labels=y_test[(1,6,7,8),],
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
                print(f"{-plot}{plot}")


# # # 
# Optimiser


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Adam:
    moment1: PyTree["Model"]
    moment2: PyTree["Model"]
    time: int
    alpha: float
    beta1: float
    beta2: float

    @staticmethod
    @jax.jit
    def init(
        model: PyTree["Model"],
        alpha: float,
        beta1: float,
        beta2: float,
    ) -> Self:
        return Adam(
            moment1=jax.tree.map(jnp.zeros_like, model),
            moment2=jax.tree.map(jnp.zeros_like, model),
            time=0,
            alpha=alpha,
            beta1=beta1,
            beta2=beta2,
        )

    @jax.jit
    def update(
        self: Self,
        grads: PyTree["Model"],
    ) -> tuple[
        PyTree["Model"],
        Self,
    ]:
        # update the optimiser state
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

        # compute the model update
        moment1_unbiased = jax.tree.map(
            lambda m1: m1 / (1-self.beta1**t),
            moment1,
        )
        moment2_unbiased = jax.tree.map(
            lambda m2: m2 / (1-self.beta2**t),
            moment2,
        )
        model_update = jax.tree.map(
            lambda m1, m2: - self.alpha * m1 / (jnp.sqrt(m2) + 1e-8),
            moment1_unbiased,
            moment2_unbiased,
        )

        return model_update, new_state


# # # 
# Architecture


@jax.jit
def scaled_tanh(x):
    return 1.7159 * jnp.tanh(0.6667 * x)


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AffineTransform:
    weights: Float[Array, "n_in n_out"]
    biases: Float[Array, "n_out"]

    def forward(
        self: Self,
        x: Float[Array, "n_in"],
    ) -> Float[Array, "n_out"]:
        return x @ self.weights + self.biases

    @staticmethod
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = 1/jnp.sqrt(num_inputs)
        return AffineTransform(
            weights=jax.random.uniform(
                key=key,
                shape=(num_inputs, num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
            biases=jnp.zeros(num_outputs),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Subsample2x2:
    weights: Float[Array, "c 1 1"]
    biases: Float[Array, "c 1 1"]

    def forward(
        self: Self,
        x: Float[Array, "c 2h 2w"],
    ) -> Float[Array, "c h w"]:
        sums = einops.reduce(
            x,
            'c (h 2) (w 2) -> c h w',
            'sum',
        )
        return sums * self.weights + self.biases

    @staticmethod
    def init(num_channels: int) -> Self:
        return Subsample2x2(
            weights=jnp.ones((num_channels, 1, 1)) / 4,
            biases=jnp.zeros((num_channels, 1, 1)),
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["kernel"],
    meta_fields=["padding"],
)
@dataclasses.dataclass
class SimpleConv2d:
    kernel: Float[Array, "k k c_in c_out"]
    padding: int

    def forward(
        self: Self,
        x: Float[Array, "c_in h w"],
    ) -> Float[Array, "c_out h_ w_"]:
        x_1chw = einops.rearrange(x, 'c h w -> 1 c h w')
        y_1chw = jax.lax.conv_general_dilated(
            lhs=x_1chw,
            rhs=self.kernel,
            window_strides=(1,1),
            padding=[
                (self.padding, self.padding),
                (self.padding, self.padding),
            ],
            dimension_numbers=("NCHW", "HWIO", "NCHW"),
        )
        y_chw = y_1chw[0]
        return y_chw

    @staticmethod
    def init(
        key: PRNGKeyArray,
        kernel_size: int,
        num_channels_in: int,
        num_channels_out: int,
        padding: int,
    ) -> Self:
        bound = 1/jnp.sqrt(num_channels_in * kernel_size**2)
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
            padding=padding,
        )
        

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

    @jax.jit
    def forward(
        self: Self,
        image: Float[Array, "28 28"],
    ) -> Float[Array, "10"]:
        x = einops.rearrange(image, 'h w -> 1 h w') # -> 1 x 28 x 28
        x = scaled_tanh(self.C1.forward(x))         # -> 6 x 28 x 28
        x = scaled_tanh(self.S2.forward(x))         # -> 6 x 14 x 14
        x = scaled_tanh(self.C3.forward(x))         # -> 16 x 10 x 10
        x = scaled_tanh(self.S4.forward(x))         # -> 16 x 5 x 5
        x = scaled_tanh(self.C5.forward(x))         # -> 120 x 1 x 1
        x = jnp.ravel(x)                            # -> 120
        x = scaled_tanh(self.F6.forward(x))         # -> 84
        x = self.Out.forward(x)                     # -> 10
        return jax.nn.softmax(x)
    
    @jax.jit
    def batch_forward(
        self: Self,
        images: Float[Array, "batch_size 28 28"],
    ) -> Float[Array, "batch_size 10"]:
        vmapped_forward = jax.vmap(self.forward)
        return vmapped_forward(images)

    @staticmethod
    @jax.jit
    def init(
        key: PRNGKeyArray,
    ) -> Self:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return SimpLeNet(
            C1=SimpleConv2d.init(
                key=k1,
                kernel_size=5,
                num_channels_in=1,
                num_channels_out=6,
                padding=2,
            ),
            S2=Subsample2x2.init(
                num_channels=6,
            ),
            C3=SimpleConv2d.init(
                key=k2,
                kernel_size=5,
                num_channels_in=6,
                num_channels_out=16,
                padding=0,
            ),
            S4=Subsample2x2.init(
                num_channels=16,
            ),
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


# # # 
# Metrics


@jax.jit
def cross_entropy(
    model: SimpLeNet,
    x: Float[Array, "h w"],
    y: int,
) -> float:
    probs = model.forward(x)
    return -jnp.log(probs[y])


@jax.jit
def batch_cross_entropy(
    model: SimpLeNet,
    xs: Float[Array, "batch_size h w"],
    ys: Int[Array, "batch_size"],
) -> float:
    vmapped_cross_entropy = jax.vmap(
        cross_entropy,
        in_axes=(None, 0, 0),
    )
    all_cross_entropies = vmapped_cross_entropy(
        model,
        xs,
        ys,
    ) # Float[Array, "batch_size"]
    return jnp.mean(all_cross_entropies)


@jax.jit
def batch_accuracy(
    model: SimpLeNet,
    xs: Float[Array, "batch_size h w"],
    ys: Int[Array, "batch_size"],
) -> float:
    pred_prob_all_classes = model.batch_forward(xs)
    highest_prob_class = pred_prob_all_classes.argmax(axis=1)
    correct_classifications = (ys == highest_prob_class)
    return jnp.mean(correct_classifications)


# # # 
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    labels: Int[Array, "n"],
    model: SimpLeNet,
) -> mp.plot:
    # shrink and normalise images
    digs = einops.reduce(
        (digits + 0.1) / 1.275,
        'b (h 2) (w 2) -> b h w',
        'mean',
    )
    width = digs.shape[-1]

    # classify digits and mark correct or incorrect
    pred_probs = model.batch_forward(digits)
    pred_labels = pred_probs.argmax(axis=-1)
    corrects = (labels == pred_labels)
    cmaps = [mp.cyans if correct else mp.magentas for correct in corrects]

    # build the visualisation
    array = mp.wrap(*[
        mp.text("p( digit | image )")
        / mp.columns(
            probs,
            height=6,
            vrange=1,
            column_width=1,
            column_spacing=1,
            colors=[mp.cyber(i==label) for i in range(10)],
        )
        / mp.text(" ".join(str(d) for d in range(10)))
        + mp.text("image")
        / mp.image(dig, colormap=cmap)
        for dig, label, probs, cmap in zip(digs, labels, pred_probs, cmaps)
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
            width=28,
            height=9,
        ),
        title=f"cross entropy {losses[-1][1]:.3f}",
        xlabel="train steps",
    )
    acc_plot = mp.axes(
        mp.scatter(
            (accuracies, 'cyan'),
            xrange=(0, total_num_steps-1),
            yrange=(0, 1),
            width=28,
            height=9,
        ),
        title=f"test accuracy {accuracies[-1][1]:.2%}",
        xlabel="train steps",
    )
    return loss_plot + acc_plot


# # # 
# Entry point

if __name__ == "__main__":
    tyro.cli(main)
