"""
CNN for handwritten digit classification, implemented with equinox and optax.

Preliminaries:

* any questions from homework?
* installations:
  * today we'll see `optax` (already installed?)
* download data!
  * same as last week `cp ../workshop3/mnist.npz .

Notes:

* 'jax deep learning ecosystem'
  https://deepmind.google/discover/blog/using-jax-to-accelerate-our-research/
* we will be avoiding eqx.filter* stuff until next time

Workshop plan: starting from code similar to last time:

1. eqx.nn: implement simplified LeNet CNN
2. jax.vmap: vectorise forward pass and cross entropy loss
3. optax: state management for optimisation

Challenge:

* implement a drop-in replacement for `optax.adam`

"""

from typing import Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray

import jax
import jax.numpy as jnp
import einops
import optax
import equinox as eqx

import tqdm
import mattplotlib as mp


# # # 
# Architecture


class Subsample2x2(eqx.Module):
    weights: Float[Array, "c 1 1"]
    biases: Float[Array, "c 1 1"]


    def __init__(self, num_channels: int):
        self.weights = jnp.ones((num_channels, 1, 1)) / 4
        self.biases = jnp.zeros((num_channels, 1, 1))


    def __call__(self, x: Float[Array, "c h w"]) -> Float[Array, "c h w"]:
        sums = einops.reduce(x, 'c (h 2) (w 2) -> c h w', 'sum')
        return self.weights * sums + self.biases


class SimpLeNet(eqx.Module):
    C1: eqx.nn.Conv2d
    S2: Subsample2x2
    C3: eqx.nn.Conv2d
    S4: Subsample2x2
    C5: eqx.nn.Conv2d
    F6: eqx.nn.Linear
    Out: eqx.nn.Linear


    def __init__(self, key: PRNGKeyArray):
        raise NotImplementedError


    def forward(
        self,
        image: Float[Array, "1 28 28"],
    ) -> Float[Array, "10"]:
        raise NotImplementedError
        # Input:         1x28x28
        # C1:       ->   6x28x28
        # S2:       ->   6x14x14
        # C3*:      ->  16x10x10 (note: fully connected channels)
        # S4:       ->  16x5x5
        # C5:       -> 120x1x1 (note: equiv. dense 400->120 at this size)
        # (flatten) -> 120
        # F6:       -> 84
        # Output*:  -> 10 (note: learned map, no hand-made RBF code)


    def forward_batch(
        self,
        x_batch: Float[Array, "b 1 28 28"],
    ) -> Float[Array, "b 10"]:
        raise NotImplementedError


def scaled_tanh(x):
    return 1.7159 * jnp.tanh(0.6667 * x)


# # # 
# Training loop


def main(
    learning_rate: float = 0.05,
    lr_schedule: bool = False,
    opt: Literal["sgd", "adam", "adamw"] = "sgd",
    batch_size: int = 512,
    num_steps: int = 128,
    steps_per_visualisation: int = 4,
    num_digits_per_visualisation: int = 10,
    seed: int = 42,
):
    key = jax.random.key(seed)


    # initialise model (same as last time)
    key_model, key = jax.random.split(key)
    model = SimpLeNet(key=key_model)

    print(model)

    
    # load and preprocess data (same as last time)
    with jnp.load('mnist.npz') as datafile:
        x_train = jnp.array(datafile['x_train'])
        x_test = jnp.array(datafile['x_test'])
        y_train = jnp.array(datafile['y_train'])
        y_test = jnp.array(datafile['y_test'])
    x_train = 1.275 * x_train / 255 - 0.1
    x_test = 1.275 * x_test / 255 - 0.1

    print(model.forward(x_train[0]))
    print(model.forward_batch(x_train[:2]))
    

    print("initialising optimiser...")
    # configure learning rate schedule
    if lr_schedule:
        pass
    # configure optimiser
    if opt == 'sgd':
        pass
    elif opt == 'adam':
        pass
    elif opt == 'adamw':
        pass
    # initialise the optimiser state
    pass
    
    print(opt_state)


    print("begin training...")
    losses = []
    accuracies = []
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
        # sample a batch (same as last time)
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
        pass


        # track metrics
        losses.append((step, loss))
        test_acc = batch_accuracy(model, x_test[:1000], y_test[:1000])
        accuracies.append((step, test_acc))


        # visualisation!
        if step % steps_per_visualisation == 0 or step == num_steps - 1:
            digit_plot = vis_digits(
                digits=x_test[:num_digits_per_visualisation],
                true_labels=y_test[:num_digits_per_visualisation],
                model=model,
            )
            metrics_plot = vis_metrics(
                losses=losses,
                accuracies=accuracies,
                total_num_steps=num_steps,
            )
            plot = digit_plot ^ metrics_plot
            if step == 0:
                tqdm.tqdm.write(str(plot))
            else:
                tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")


# # # 
# Metrics


def batch_cross_entropy(
    model: SimpLeNet,
    x_batch: Float[Array, "b h w"],
    y_batch: Int[Array, "b"],
) -> float:
    raise NotImplementedError


def cross_entropy(
    model: SimpLeNet,
    x: Float[Array, "h w"],
    y: int,
) -> float:
    # Cross entropy formula: Hx(q, p) = - Sum_i p(i) log q(i)
    raise NotImplementedError


# def cross_entropy(
#     model: MLPImageClassifier,
#     x_batch: Float[Array, "b h w"],
#     y_batch: Int[Array, "b"],
# ) -> float:
#     batch_size, = y_batch.shape
#     pred_prob_all_classes = model(x_batch)          # -> batch_size 10
#     pred_prob_true_class = pred_prob_all_classes[   # advanced indexing
#         jnp.arange(batch_size),                     # for each example
#         y_batch,                                    # select prob of true class
#     ]                                               # -> batch_size
#     log_prob_true_class = jnp.log(pred_prob_true_class)
#     avg_cross_entropy = -jnp.mean(log_prob_true_class)
#     return avg_cross_entropy


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
    model: SimpLeNet | None = None,
) -> mp.plot:
    # shrink and normalise images
    ddigits = einops.reduce(
        (digits + 0.1) / 1.275,
        'b (h 2) (w 2) -> b h w',
        'mean',
    )
    width = ddigits.shape[-1]

    # if model is provided, classify digits and mark correct or incorrect
    if model is not None:
        pred_labels = model.forward_batch(digits).argmax(axis=-1)
        corrects = (true_labels == pred_labels)
        cmaps = [None if correct else mp.reds for correct in corrects]
        labels = [f"{t} ({p})" for t, p in zip(true_labels, pred_labels)]
    else:
        cmaps = [None] * len(true_labels)
        labels = [str(t) for t in true_labels]
    array = mp.wrap(*[
        mp.border(
            mp.image(ddigit, colormap=cmap)
            ^ mp.center(mp.text(label), width=width)
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
            height=11,
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
            height=11,
        ))
        ^ mp.text(f"acc: {accuracies[-1][1]:.2%}")
    )
    return loss_plot & acc_plot


# # # 
# Entry point

if __name__ == "__main__":
    import tyro
    tyro.cli(main)
