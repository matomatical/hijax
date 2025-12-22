"""
Lecture 10: Hi, algorithms!

Demonstration: Comparative implementation of Kruskall's minimum spanning tree
algorithm with different union-find data structures.

Learning objectives:

* further exploration of jax.jit, branching computations
* jax.lax.while
* performance considerations of branching computations
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
    size: int = 25,
    seed: int = 42,
):
    key = jax.random.key(seed=seed)

    # initialise environment
    key_init, key = jax.random.split(key)
    state = Environment.init(
        key=key_init,
        size=size,
    )
    print(mp.image(state.render()))

    print("doing random walks...")
    key_walks, key = jax.random.split(key)
    state_bt = jax.vmap(
        walk,
        in_axes=(0, None, None),
    )(
        jax.random.split(key_walks, 256),
        size,
        512,
    )

    print("saving...")
    save_animation(state_bt)


# # # 
# Environment


class Action(enum.IntEnum):
    WAIT = 0
    UP = 1
    LEFT = 2
    DOWN = 3
    RIGHT = 4


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Environment:
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    walls: Bool[Array, "size size"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["size"])
    def init(
        key: PRNGKeyArray,
        size: int,
    ) -> Self:
        # generate random wall layout
        # TODO

        # generate random hero and goal positions
        # TODO

        # ensure these positions are free of walls
        walls = walls.at[hero_pos[0], hero_pos[1]].set(False)
        walls = walls.at[goal_pos[0], goal_pos[1]].set(False)

        return Environment(
            hero_pos=hero_pos,
            goal_pos=goal_pos,
            walls=walls,
        )

    @jax.jit
    def render(self: Self) -> Float[Array, "size size 3"]:
        size = self.walls.shape[0]
        rgb = jnp.zeros((size, size, 3))
        rgb = jnp.where(
            self.walls[:, :, None],
            jnp.array([.2, .2, .4]),
            rgb,
        )
        rgb = rgb.at[
            self.hero_pos[0],
            self.hero_pos[1],
        ].set(jnp.array((0,1,1)))
        rgb = rgb.at[
            self.goal_pos[0],
            self.goal_pos[1],
        ].set(jnp.array((1,0,1)))
        return rgb

    @jax.jit
    def step(
        self: Self,
        action: Action,
    ) -> Self:
        # move the hero
        deltas = jnp.array((
            ( 0,  0),
            (-1,  0),
            ( 0, -1),
            (+1,  0),
            ( 0, +1),
        ))
        new_hero_pos = self.hero_pos + deltas[action]

        # check for collisions
        collision = self.walls[
            new_hero_pos[0],
            new_hero_pos[1],
        ]
        new_hero_pos = jnp.where(
            collision,
            self.hero_pos,
            new_hero_pos,
        )

        return dataclasses.replace(self, hero_pos=new_hero_pos)

    @jax.jit
    def goal(self: Self) -> bool:
        return jnp.all(self.hero_pos == self.goal_pos)


@functools.partial(jax.jit, static_argnames=["size", "num_steps"])
def walk(
    key: PRNGKeyArray,
    size: int,
    num_steps: int,
) -> Environment: # Environment[time]
    # initialise environment
    key_init, key = jax.random.split(key)
    state = Environment.init(
        key=key_init,
        size=size,
    )

    def step(carry, _input):
        key, state = carry
        # take a step
        # TODO: TAKE VALID STEPS ONLY
        key_step, key = jax.random.split(key)
        action = jax.random.choice(
            key=key_step,
            a=5,
            shape=(),
        )
        state = state.step(action=action)
        # reset if goal hit
        reset = state.goal()
        key_reset, key = jax.random.split(key)
        new_state = Environment.init(
            key=key_reset,
            size=size,
        )
        next_state = jax.tree.map(
            lambda old, new: jnp.where(reset, new, old),
            state,
            new_state,
        )
        return (key, next_state), next_state

    _final_carry, next_states = jax.lax.scan(
        step,
        (key, state),
        length=num_steps,
    )

    return jax.tree.map(
        lambda l, ls: jnp.concatenate((l[None], ls), axis=0),
        state,
        next_states,
    )


# # # 
# Maze generator


# TODO


# # # 
# Export to gif


def save_animation(
    state_bt: Environment, # Environment[batch time]
    filename="output.gif",
):
    imgss = jax.vmap(jax.vmap(Environment.render))(state_bt)
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
