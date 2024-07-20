"""
MLP for handwritten digit classification, implemented with equinox.

Preliminaries:

* any questions from homework?
* installations:
  * new library! `pip install equinox`
  * new library! `pip install jaxtyping`
  * today we'll also see `einops` and `mattplotlib`
* download data! MNIST (11MB)
  ```
  curl https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz --output mnist.npz
  ```

Notes:

* options for deep learning in jax (try not to rant)
  * deepmind's `haiku`
  * google brain's `flax.linen`
  * patrick kidger's `equinox` (most pedagogically convenient)
  * google deepmind's `flax.nnx` (next generation by fiat)
* new jax concept: pytrees
  * we saw these last time actually

Workshop plan:

* implement image classifier MLP with equinox
* load MNIST data set
* train MLP on MNIST with minibatch SGD

Challenge:

* manually register the MLP modules as pytrees (obviating equinox dependency)
"""

from typing import Literal
from jaxtyping import Array, Float, Int, PRNGKeyArray as Key

import jax
import jax.numpy as jnp
import einops
import equinox

import tqdm
import mattplotlib as mp


# # # 
# Architecture


# TODO


# # # 
# Training loop


def main(
    num_hidden: int = 300,
    learning_rate: float = 0.05,
    batch_size: int = 512,
    num_steps: int = 256,
    steps_per_visualisation: int = 4,
    num_digits_per_visualisation: int = 15,
    seed: int = 42,
):
    key = jax.random.key(seed)


    # initialise model
    print("initialising model...")
    print("TODO.")


    print("loading and preprocessing data...")
    print("TODO.")


    print("begin training...")
    losses = []
    accuracies = []
    for step in tqdm.trange(num_steps, dynamic_ncols=True):
        # sample a batch
        tqdm.tqdm.write("TODO: sample a batch...")

        # compute the batch loss and grad
        tqdm.tqdm.write("TODO: compute batch loss and grad...")

        # update model
        tqdm.tqdm.write("TODO: update model (and optimiser)...")

        break

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
            plot = digit_plot ^ metrics_plot
            if step == 0:
                tqdm.tqdm.write(str(plot))
            else:
                tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")


# # # 
# Metrics


# TODO


# # # 
# Visualisation


def vis_digits(
    digits: Float[Array, "n h w"],
    true_labels: Int[Array, "n"],
    pred_labels: Int[Array, "n"] | None = None,
) -> mp.plot:
    # downsample images
    ddigits = digits[:,::2,::2]
    dwidth = digits.shape[-1] // 2

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
