"""
Lecture 08: Hi, static arguments!

Demonstration: Implement and train a byte transformer on the Sherlock Holmes
canon.

Learning objectives:

* further exploration of jax.jit (jit dojo lessons 4 and 5):
  * static arguments and valid types for static arguments
  * static fields in data classes
* dynamic slicing
"""

import dataclasses
import functools
import tyro
import tqdm
import matthewplotlib as mp

from typing import Self
from jaxtyping import Array, Float, Int, UInt8 as Byte, PRNGKeyArray, PyTree

import numpy as np
import jax
import jax.numpy as jnp
import einops


def main(
    # model config
    num_blocks: int = 6,
    num_heads: int = 8,
    embed_size: int = 256,
    mlp_size: int = 256,
    max_context_length: int = 64,
    completion_length: int = 256,
    # training config
    learning_rate: float = 0.0001,
    batch_size: int = 32,
    num_steps: int = 2048,
    num_steps_per_reset: int = 32,
    seed: int = 221,
):
    key = jax.random.key(seed=seed)

    
    print("loading byte corpus...")
    

    print("configuring model architecture...")


    print("testing model completion...")

    
    print("initialising optimiser...")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
    )


    print("training loop...")
 
 
# # # 
# Helper functions


# # # 
# Architecture


# # # 
# Cross entropy functions


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
        beta1: float = 0.9,
        beta2: float = 0.999,
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
# Visualisation


def wrap(text: str, max_width: int, max_lines: int) -> str:
    num_lines = min(len(text) // max_width + 1, max_lines)
    lines = [
        text[i:i+max_width]
        for i in range(0, num_lines*max_width, max_width)
    ]
    return "\n".join(lines)


def vis_example(
    prompt: str,
    completion: str,
    t: int,
    T: int,
) -> mp.plot:
    # strip quotes

    # prompt
    render_prompt = repr(prompt)[1:-1]
    wrapped_prompt = wrap(render_prompt, max_width=29, max_lines=13)
    plot_prompt = mp.text(wrapped_prompt, fgcolor="cyan")
    
    # completion
    render_completion = repr(completion)[1:-1]
    offset_completion = " "*len(render_prompt) + render_completion
    wrapped_completion = wrap(offset_completion, max_width=29, max_lines=13)
    plot_completion = mp.text(wrapped_completion, fgcolor='magenta')

    return mp.border(
        mp.dstack(
            mp.blank(width=29, height=13),
            plot_completion,
            plot_prompt,
        ),
        title=f"completion at {t:4d}/{T:4d}",
    )


if __name__ == "__main__":
    tyro.cli(main)
