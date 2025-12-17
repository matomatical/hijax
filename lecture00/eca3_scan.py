import time
import jax
import jax.numpy as jnp
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

    compiled_simulate = jax.jit(
        simulate,
        static_argnames=["width", "num_steps"],
    )

    start_time = time.perf_counter()
    states = compiled_simulate(
        rule=rule,
        width=width,
        num_steps=num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("compile complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")
    
    start_time = time.perf_counter()
    states = compiled_simulate(
        rule=rule,
        width=width,
        num_steps=num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("simulation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")

    if print_image:
        print(mp.image((0.2 + states) / 1.2))

    if save_image:
        print("rendering to 'output.png'...")
        Image.fromarray(255 * np.array(states)).save('output.png')


def simulate(
    rule: UInt8[Array, ""],
    width: int,
    num_steps: int,
) -> UInt8[Array, "num_steps width"]:
    # parse rule
    rule_uint8 = jnp.uint8(rule)
    rule_bits = jnp.unpackbits(rule_uint8, bitorder='little')
    rule_table = rule_bits.reshape(2,2,2)
    
    # initialise state
    state: UInt8[Array, "width"]
    state = jnp.zeros(width, dtype=jnp.uint8)
    state = state.at[width//2].set(1)

    # simulate
    def step(state, _):
        state_wrapped: UInt8[Array, "width+2"]
        state_wrapped = jnp.pad(state, 1, mode='wrap')
        state = rule_table[
            state_wrapped[0:-2],
            state_wrapped[1:-1],
            state_wrapped[2:],
        ]
        return state, state
    _final_state, next_states = jax.lax.scan(
        step,
        state,
        length=num_steps-1,
    )
    
    return jnp.concatenate([state[None], next_states], axis=0)


if __name__ == "__main__":
    tyro.cli(main)
