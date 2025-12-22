"""
Lecture 09: Hi, branching computation!

Demonstration: Implement a simple grid-world maze environment.

Learning objectives:

* further exploration of jax.jit (jit dojo):
  * if statements are frozen at trace time
* workarounds:
  * jax.lax.cond and jax.lax.select primitives
  * jax.numpy.where
  * indexing
  * boolean multiplication
"""

import dataclasses
import enum
import functools
import time
import tyro
import matthewplotlib as mp
from PIL import Image

from jaxtyping import Int, Bool, Float, Array, PRNGKeyArray
from typing import Self

import numpy as np
import jax
import jax.numpy as jnp
import einops


# # # 
# Entry point


def main(
    size: int = 12,
    seed: int = 42,
):
    # TODO


# # # 
# Export to gif


def save_animation(
    statess: Environment, # Environment[batch time]
    filename="output.gif",
):
    imgss = jax.vmap(jax.vmap(Environment.render))(statess)
    imgss = jnp.pad(
        imgss,
        pad_width=(
            (0, 0), # env
            (0, 0), # steps
            (0, 1), # height
            (0, 1), # width
            (0, 0), # channel
        ),
    )
    grid = einops.rearrange(
        imgss,
        '(H W) t h w c -> t (H h) (W w) c',
        H=16,
        W=16,
    )
    grid = jnp.pad(
        grid,
        pad_width=(
            (0, 4), # time
            (1, 0), # height
            (1, 0), # width
            (0, 0), # channel
        ),
    )
    frames = np.asarray(grid * 255, dtype=np.uint8)
    Image.fromarray(frames[0]).save(
        filename,
        format="gif",
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=20,
        loop=0,
    )


if __name__ == "__main__":
    tyro.cli(main)
