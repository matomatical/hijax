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

from jaxtyping import Int, Bool, Float, Array, PRNGKeyArray
from typing import Self

import numpy as np
import jax
import jax.numpy as jnp
import einops
import tyro
import matthewplotlib as mp
from PIL import Image


# # # 
# Environment class


class Action(enum.IntEnum):
    WAIT = 0 # do nothing
    UP = 1 # move up
    LEFT = 2 # move left
    DOWN = 3 # move down
    RIGHT = 4 # move right


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Environment:
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    walls: Bool[Array, "size size"]

    @functools.partial(jax.jit, static_argnames=("size",))
    def init(key: PRNGKeyArray, size: int) -> Self:
        # generate random wall layout
        walls = jnp.zeros((size, size), dtype=bool)
        walls = walls.at[(0,-1),:].set(True)
        walls = walls.at[:,(0,-1)].set(True)
        key_walls, key = jax.random.split(key)
        inner_walls = jax.random.bernoulli(
            key=key_walls,
            p=0.1,
            shape=(size-2, size-2),
        )
        walls = walls.at[1:-1,1:-1].set(inner_walls)

        # generate random hero and goal positions
        key_pos, key = jax.random.split(key)
        indices = jax.random.choice(
            key=key_pos,
            a=(size-2)**2,
            shape=(2,),
            replace=False,
        )
        div, mod = jnp.divmod(indices, size-2)
        hero_pos = 1 + jnp.array([div[0], mod[0]])
        goal_pos = 1 + jnp.array([div[1], mod[1]])

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
            self.walls[:,:,None],
            jnp.array([.2,.2,.4]),
            rgb,
        )
        rgb = rgb.at[
            self.hero_pos[0],
            self.hero_pos[1],
        ].set(jnp.array([0,1,1]))
        rgb = rgb.at[
            self.goal_pos[0],
            self.goal_pos[1],
        ].set(jnp.array([1,0,1]))
        return rgb

    @jax.jit
    def step(self: Self, action: Action) -> Self:
        # move hero
        deltas = jnp.array([
            ( 0,  0), # stay
            (-1,  0), # move up
            ( 0, -1), # move left
            (+1,  0), # move down
            ( 0, +1), # move right
        ])
        try_hero_pos = self.hero_pos + deltas[action]
        collision = self.walls[
            try_hero_pos[0],
            try_hero_pos[1],
        ]
        new_hero_pos = jnp.where(
            collision,
            self.hero_pos,
            try_hero_pos,
        )
        self = dataclasses.replace(self, hero_pos=new_hero_pos)
        return self


    @jax.jit
    def goal(self: Self) -> bool:
        return jnp.all(self.hero_pos == self.goal_pos)
        
    
@functools.partial(jax.jit, static_argnames=("size", "length",))
def walk(
    key: PRNGKeyArray,
    size: int,
    length: int,
) -> Environment: # Environment[length]
    # define the initial carry (key, state)
    key_init, key = jax.random.split(key)
    initial_state = Environment.init(key=key_init, size=size)
    initial_carry = (key, initial_state)

    # define a step of the random walk
    def step(carry, input_):
        key, state = carry
        # take a step
        # TODO: Conditional random policy, another good exercise
        key_step, key = jax.random.split(key)
        action = jax.random.choice(
            key=key_step,
            a=len(Action),
            shape=(),
        )
        state = state.step(action)
        # reset if hit goal
        key_reset, key = jax.random.split(key)
        new_state = Environment.init(key=key_reset, size=size)
        reset = state.goal()
        state = jax.tree.map(
            lambda old, new: jnp.where(reset, new, old),
            state,
            new_state,
        )
        # output
        return (key, state), state

    # scan the walk
    final_carry, tail_states = jax.lax.scan(
        step,
        initial_carry,
        length=length-1,
    )
    all_states = jax.tree.map(
        lambda x, xs: jnp.concatenate([x[None], xs], axis=0),
        initial_state,
        tail_states,
    )
    return all_states


# # # 
# Entry point


def main(
    size: int = 12,
    seed: int = 42,
):
    key = jax.random.key(seed=seed)

    print("initialise environment...")
    key_init, key = jax.random.split(key)
    state = Environment.init(key=key_init, size=size)
    print(mp.image(state.render()))


    print("random walk...")
    print(mp.image(state.render()))
    while False:
        # take a step
        key_step, key = jax.random.split(key)
        action = jax.random.choice(
            key=key_step,
            a=len(Action),
            shape=(),
        )
        state = state.step(action)

        # reset if hit goal
        key_reset, key = jax.random.split(key)
        new_state = Environment.init(key=key_reset, size=size)
        reset = state.goal()
        state = jax.tree.map(
            lambda old, new: jnp.where(reset, new, old),
            state,
            new_state,
        )

        plot = mp.image(state.render())
        print(f"{-plot}{plot}")
        time.sleep(0.02)

    print("scanned and vmapped random walk...")
    statess = jax.vmap(walk, in_axes=(0,None,None))(
        jax.random.split(key, 256),
        12,
        512,
    )
    print("done!")
    print("rendering...")
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
        "../gallery/lecture09.gif",
        format="gif",
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=20,
        loop=0,
    )


if __name__ == "__main__":
    tyro.cli(main)
