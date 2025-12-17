import time
import jax
import jax.numpy as jnp
import numpy as np
import einops
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

    vmapped_simulate = jax.vmap(
        simulate,
        in_axes=(0, None, None),
        out_axes=0,
    )

    compiled_simulate = jax.jit(
        vmapped_simulate,
        static_argnames=["width", "num_steps"],
    )

    rules = jnp.arange(256)

    start_time = time.perf_counter()
    states = compiled_simulate(
        rules,
        width,
        num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("compile complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")
    
    start_time = time.perf_counter()
    states = compiled_simulate(
        rules,
        width,
        num_steps,
    ).block_until_ready()
    end_time = time.perf_counter()
    print("simulation complete!")
    print(f"time taken {end_time - start_time:.5f} seconds")

    if print_image:
        for rule, rule_states in enumerate(states):
            print("rule", rule)
            print(mp.image((0.2 + rule_states) / 1.2))
            time.sleep(0.5)

    if save_image:
        print("rendering to 'output.png'...")
        formatted_states = einops.rearrange(
            np.array(states),
            '(h w) num_steps width -> (h num_steps) (w width)',
            h=16,
            w=16,
        )
        Image.fromarray(255 * formatted_states).save('output.png')


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
