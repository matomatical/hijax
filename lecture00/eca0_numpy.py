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
import tyro
from jaxtyping import Array, UInt8, Bool
import matthewplotlib as mp
from PIL import Image


def main(
    rule: int = 110,
    width: int = 80,
    num_steps: int = 80,
    print_image: bool = True,
    save_image: bool = False,
):
    start_time = time.perf_counter()
    states = simulate(
        rule=rule,
        width=width,
        num_steps=num_steps,
    )
    end_time = time.perf_counter()
    print("simulation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")

    if print_image:
        print(mp.image((0.2 + states) / 1.2))

    if save_image:
        print("rendering to 'output.png'...")
        Image.fromarray(255 * states).save('output.png')


def simulate(
    rule: UInt8[Array, ""],
    width: int,
    num_steps: int,
) -> UInt8[Array, "num_steps width"]:
    # parse rule
    rule_uint8 = np.uint8(rule)
    print(f"rule: {rule_uint8:3d} (0b{rule_uint8:08b})")
    
    rule_bits = np.unpackbits(rule_uint8, bitorder='little')
    print("bits:", rule_bits)
    
    rule_table = rule_bits.reshape(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                print(f"rule_table[{i},{j},{k}] = {rule_table[i,j,k]}")
    
    # initialise state
    state: UInt8[Array, "width"]
    state = np.zeros(width, dtype=np.uint8)
    state[width//2] = 1

    # simulate
    states = [state]
    for t in range(num_steps-1):
        state_wrapped: UInt8[Array, "width+2"]
        state_wrapped = np.pad(state, 1, mode='wrap')
        state = rule_table[
            state_wrapped[0:-2],
            state_wrapped[1:-1],
            state_wrapped[2:],
        ]
        states.append(state)
    
    return np.stack(states)


if __name__ == "__main__":
    tyro.cli(main)
