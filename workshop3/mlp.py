"""
MLP for handwritten digit classification, implemented with equinox and optax.

Workshop plan:

* any questions from homework?
* installations:
  * new library! `pip install equinox`
  * today we'll also embrace `jaxtyping`, `einops`, and `mattplotlib`
* deep learning in jax:
  * dm-haiku, flax.linen, equinox, flax.nnx, try not to rant too much
  * optax and the 'dm jax ecosystem'
* new jax concept: pytrees (we saw these last time)
* workshop 3 demo:
  * implement MLP with equinox
  * load MNIST https://yann.lecun.com/exdb/mnist/
  * train MLP on MNIST with optax (sgd, adam, lr schedule, adamw)
* challenges:
  * implement a stateful optimiser object from scratch
  * replicate some architectures and performance numbers from lecun's table
"""

from jaxtyping import Array, Float, Int, PRNGKeyArray as Key
from typing import Literal

import jax
import jax.numpy as jnp
import einops
import optax
import equinox

import tqdm
import draft_mattplotlib as mp


# # # 
# Architecture


class LinearLayer(equinox.Module):
    weight_matrix: Array
    bias_vector: Array

    def __init__(
        self,
        key: Key,
        num_inputs: int,
        num_outputs: int,
    ):
        # Xavier-initialised weight matrix
        init_bound = jnp.sqrt(6/(num_inputs + num_outputs))
        self.weight_matrix = jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-init_bound,
            maxval=init_bound,
        )

        # zero-initialised bias vector
        self.bias_vector = jnp.zeros((num_outputs,))

    def __call__(
        self,
        x: Float[Array, '... num_inputs'],
    ) -> Float[Array, '... num_outputs']:
        return x @ self.weight_matrix + self.bias_vector


class MLPImageClassifier(equinox.Module):
    layer1: LinearLayer
    layer2: LinearLayer

    def __init__(
        self,
        key: Key,
        image_shape: tuple[int, int],
        num_hidden: int,
        num_classes: int,
    ):
        key_layer1, key_layer2 = jax.random.split(key)
        num_inputs = image_shape[0] * image_shape[1]
        self.layer1 = LinearLayer(key_layer1, num_inputs, num_hidden)
        self.layer2 = LinearLayer(key_layer2, num_hidden, num_classes)

    def __call__(
        self,
        x: Float[Array, '... image_height image_width'],
    ) -> Float[Array, '... num_outputs']:
        # flatten image
        x = einops.rearrange(x, '... h w -> ... (h w)')
        # apply mlp
        x = self.layer1(x)
        x = jnp.tanh(x)
        x = self.layer2(x)
        # logits -> probability distribution
        x = jax.nn.softmax(x, axis=-1)
        return x


# # # 
# Training loop


def main(
    num_hidden: int = 300,
    learning_rate: float = 0.05,
    lr_schedule: bool = False,
    opt: Literal["sgd", "adam", "adamw"] = "sgd",
    batch_size: int = 512,
    num_steps: int = 256,
    steps_per_visualisation: int = 4,
    num_digits_per_visualisation: int = 15,
    seed: int = 42,
):
    key = jax.random.key(seed)

    # initialise model
    print("initialising model...")
    key_model, key = jax.random.split(key)
    model = MLPImageClassifier(
        key=key_model,
        image_shape=(28, 28),
        num_hidden=num_hidden,
        num_classes=10,
    )

    # print(model)
    # print(model(jnp.zeros((2,28,28))))

    print("loading and preprocessing data...")
    with jnp.load('mnist.npz') as datafile:
        x_train = datafile['x_train']
        x_test = datafile['x_test']
        y_train = datafile['y_train']
        y_test = datafile['y_test']
    x_train, x_test, y_train, y_test = jax.tree.map(
        jnp.array,
        (x_train, x_test, y_train, y_test),
    )
    x_train, x_test = jax.tree.map(
        lambda x: x/255,
        (x_train, x_test),
    )

    # print(vis_digits(
    #     digits=x_test[:num_digits_per_visualisation],
    #     true_labels=y_test[:num_digits_per_visualisation],
    #     # pred_labels=model(
    #     #     x_test[:num_digits_per_visualisation]
    #     # ).argmax(axis=-1),
    # ))

    print("initialising optimiser...")
    # configure the optimiser
    if lr_schedule:
        learning_rate = optax.linear_schedule(
            init_value=learning_rate,
            end_value=learning_rate/10,
            transition_steps=num_steps,
        )
    if opt == 'sgd':
        optimiser = optax.sgd(learning_rate)
    elif opt == 'adam':
        optimiser = optax.adam(learning_rate)
    elif opt == 'adamw':
        optimiser = optax.adamw(learning_rate)
    # initialise the optimiser state
    opt_state = optimiser.init(model)
    
    # print(opt_state)

    print("begin training...")
    losses = []
    accuracies = []
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
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
        loss, grads = jax.value_and_grad(cross_entropy)(
            model,
            x_batch,
            y_batch,
        )

        # compute update, update optimiser and model
        updates, opt_state = optimiser.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)

        # track metrics
        losses.append((step, loss))
        test_acc = accuracy(model, x_test[:1000], y_test[:1000])
        accuracies.append((step, test_acc))

        # visualisation!
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            digit_plot = vis_digits(
                digits=x_test[:num_digits_per_visualisation],
                true_labels=y_test[:num_digits_per_visualisation],
                pred_labels=model(
                    x_test[:num_digits_per_visualisation]
                ).argmax(axis=-1),
            )
            metrics_plot = vis_metrics(
                losses=losses,
                accuracies=accuracies,
                total_num_steps=num_steps,
            )
            opt_state_str = str(opt_state)
            output_height = (
                digit_plot.height
                + metrics_plot.height
                + 1+len(opt_state_str.splitlines())
            )

            tqdm.tqdm.write(
                (f"\x1b[{output_height}A" if step > 0 else "")
                + f"{digit_plot}\n"
                + f"{metrics_plot}\n"
                + f"optimiser state:\n{opt_state_str}"
            )


def cross_entropy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    """
    Hx(q, p) = - Sum_i p(i) log q(i)
    """
    batch_size, = y_batch.shape
    pred_prob_all_classes = model(x_batch)          # -> batch_size 10
    pred_prob_true_class = pred_prob_all_classes[   # advanced indexing
        jnp.arange(batch_size),                     # for each example
        y_batch,                                    # select prob of true class
    ]                                               # -> batch_size
    log_prob_true_class = jnp.log(pred_prob_true_class)
    avg_cross_entropy = -jnp.mean(log_prob_true_class)
    return avg_cross_entropy


def accuracy(
    model: MLPImageClassifier,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    pred_prob_all_classes = model(x_batch)
    highest_prob_class = pred_prob_all_classes.argmax(axis=-1)
    return jnp.mean(y_batch == highest_prob_class)


# # # 
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    true_labels: Int[Array, "n"],
    pred_labels: Int[Array, "n"] | None = None,
) -> mp.plot:
    # downsample images
    ddigits = digits[:,::2,::2]
    dwidth = digits.shape[2] // 2

    # if predictions provided, classify as true or false
    if pred_labels is not None:
        corrects = (true_labels == pred_labels)
        cmaps = [None if correct else mp.reds for correct in corrects]
        labels = [f"{t} ({p})" for t, p in zip(true_labels, pred_labels)]
    else:
        cmaps = [None] * len(true_labels)
        labels = [str(t) for t in true_labels]
    array = mp.wrap(*[
        mp.border(
            mp.image(ddigit, colormap=cmap)
            ^ mp.center(mp.text(label), width=dwidth)
        )
        for ddigit, label, cmap in zip(ddigits, labels, cmaps)
    ], cols=5)
    return array


def vis_metrics(
    losses: list[tuple[int, float]],
    accuracies: list[tuple[int, float]],
    total_num_steps: int,
) -> mp.plot:
    loss_plot = (
        mp.center(mp.text("train loss (cross entropy)"), width=40)
        ^ mp.border(mp.scatter(
            data=losses,
            xrange=(0, total_num_steps-1),
            yrange=(0, max(l for s, l in losses)),
            color=(1,0,1),
            width=38,
            height=14,
        ))
        ^ mp.text(f"loss: {losses[-1][1]:.3f}")
    )
    acc_plot = (
        mp.center(mp.text("test accuracy"), width=40)
        ^ mp.border(mp.scatter(
            data=accuracies,
            xrange=(0, total_num_steps-1),
            yrange=(0, 1),
            color=(0,1,0),
            width=38,
            height=14,
        ))
        ^ mp.text(f"acc: {accuracies[-1][1]:.2%}")
    )
    return loss_plot & acc_plot


# # # 
# Entry point

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
