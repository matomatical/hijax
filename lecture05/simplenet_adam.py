"""
Lecture 04: Hi, automatic vectorisation!

Demonstration: Implement and train a CNN on MNIST with minibatch SGD.

Learning objectives:

* more practice with PyTrees
* introducing jax.vmap
"""

import functools
import dataclasses
# import time # unused
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
    key = jax.random.key(seed=seed)
    
    print("load the training data...")
    # TODO

    print("initialising the model...")
    # TODO

    print("training...")
    # TODO
    

# # # 
# Architecture


class SimpLeNet:
    """TODO"""


# # # 
# Metrics

# TODO


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
