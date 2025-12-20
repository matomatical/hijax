"""
Lecture 00: Hi, JAX! How's life?

Demonstration: Port an elementary cellular automaton simulator from NumPy to
JAX.

Learning objectives:

1. Introduction to jax.numpy (jnp)
2. First look at some JAX function transformations (jit, vmap)
3. Exploration of performance aspects of NumPy, JAX, JIT
"""

import time
import numpy as np
import jax
import jax.numpy as jnp
import tyro
from jaxtyping import Array, UInt8, Bool
import matthewplotlib as mp
from PIL import Image
import einops


def main(
    rule: int = 110,
    width: int = 80,
    num_steps: int = 80,
    print_image: bool = True,
    save_image: bool = False,
):
    start_time = time.perf_counter()
    vectorised_simulate = jax.vmap(
        simulate,
        in_axes=(0, None, None),
        out_axes=0,
    )
    compiled_simulate = jax.jit(
        vectorised_simulate,
        static_argnames=["width", "num_steps"],
    )
    states = compiled_simulate(
        jnp.arange(256),
        width,
        num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("compilation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")

    start_time = time.perf_counter()
    states_rts = compiled_simulate(
        jnp.arange(256),
        width,
        num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("simulation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")

    if print_image:
        for states_ts in states_rts:
            print(mp.image((1.2 - states_ts) / 1.2))
            time.sleep(0.5)

    if save_image:
        print("rendering to 'output.png'...")
        states_grid = einops.rearrange(
            states_rts,
            '(rule_h rule_w) time state -> (rule_h time) (rule_w state)',
            rule_h=16,
            rule_w=16,
        )
        numpy_states_grid = np.asarray(states_grid)
        Image.fromarray(255 - 255 * numpy_states_grid).save('output.png')


def simulate(
    rule: UInt8[Array, ""],
    width: int,
    num_steps: int,
) -> UInt8[Array, "num_steps width"]:
    # parse rule
    rule_uint8 = jnp.uint8(rule)
    # print(f"rule: {rule_uint8:3d} (0b{rule_uint8:08b})")
    
    rule_bits = jnp.unpackbits(rule_uint8, bitorder='little')
    # print("bits:", rule_bits)
    
    rule_table = rule_bits.reshape(2,2,2)
    # for i in range(2):
    #     for j in range(2):
    #         for k in range(2):
    #             print(f"rule_table[{i},{j},{k}] = {rule_table[i,j,k]}")
    
    # initialise state
    initial_state: UInt8[Array, "width"]
    initial_state = jnp.zeros(width, dtype=jnp.uint8)
    initial_state = initial_state.at[width//2].set(1)

    # simulate
    def step(state, _) -> tuple:
        state_wrapped: UInt8[Array, "width+2"]
        state_wrapped = jnp.pad(state, 1, mode='wrap')
        next_state = rule_table[
            state_wrapped[0:-2],
            state_wrapped[1:-1],
            state_wrapped[2:],
        ]
        return next_state, next_state
    _, next_states = jax.lax.scan(
        step,
        initial_state,
        length=num_steps-1,
    )
    
    return jnp.concatenate(
        [initial_state[None], next_states],
        axis=0,
    )


if __name__ == "__main__":
    tyro.cli(main)
