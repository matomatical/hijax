import time
import numpy as np
import tyro
from jaxtyping import Array, UInt8, Bool


def main(
    rule: int = 110,
    width: int = 32,
    num_steps: int = 32,
):
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
    state[width//2] = True
    vis(state)

    # simulate
    start_time = time.perf_counter()
    for t in range(num_steps-1):
        state_wrapped: UInt8[Array, "width+2"]
        state_wrapped = np.pad(state, 1, mode='wrap')
        state = rule_table[
            state_wrapped[0:-2],
            state_wrapped[1:-1],
            state_wrapped[2:],
        ]
        vis(state)
    end_time = time.perf_counter()

    print("simulation complete!")
    print(f"time taken {end_time - start_time:.4f} seconds")


def vis(state: UInt8[Array, "width"]):
    print(''.join(["░░" if b else "██" for b in state]))

if __name__ == "__main__":
    tyro.cli(main)
