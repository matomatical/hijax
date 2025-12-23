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
    start_time = time.perf_counter()
    key_walks, key = jax.random.split(key)
    state_bt = jax.vmap(
        walk,
        in_axes=(0, None, None),
    )(
        jax.random.split(key_walks, 256),
        size,
        512,
    )
    state_bt.hero_pos.block_until_ready()
    state_bt.goal_pos.block_until_ready()
    state_bt.walls.block_until_ready()
    end_time = time.perf_counter()
    print("elapsed time", end_time - start_time, "seconds")

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
        key_walls, key = jax.random.split(key)
        walls = generate_maze(
            key=key_walls,
            size=size,
        )

        # generate random hero and goal positions
        key_pos, key = jax.random.split(key)
        indices = jax.random.choice(
            key=key_pos,
            a=(size//2)**2,
            shape=(2,),
            replace=False,
        )
        divs, mods = jnp.divmod(indices, size//2)
        hero_pos = 1 + 2 * jnp.array([divs[0], mods[0]])
        goal_pos = 1 + 2 * jnp.array([divs[1], mods[1]])

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
        # find valid steps
        probes = state.hero_pos + jnp.array((
            ( 0,  0),
            (-1,  0), 
            ( 0, -1),
            (+1,  0),
            ( 0, +1),
        ))
        valid = ~state.walls[probes[:,0], probes[:,1]]
        valid = valid.at[0].set(False)
        # take a step
        key_step, key = jax.random.split(key)
        action = jax.random.choice(
            key=key_step,
            a=5,
            shape=(),
            p=valid,
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


@functools.partial(jax.jit, static_argnames=["size"])
def generate_maze(
    key: PRNGKeyArray,
    size: int, # >= 3 and odd
) -> Bool[Array, "size size"]:
    # grid graph nodes
    S = size // 2
    nodes = jnp.arange(S**2)
    ngrid = nodes.reshape((S, S))

    # grid graph edges
    h_edges = jnp.stack((
        ngrid[:,:-1].flatten(),
        ngrid[:,1:].flatten(),
    )) # int[2, n-S]
    v_edges = jnp.stack((
        ngrid[:-1,:].flatten(),
        ngrid[1:,:].flatten(),
    )) # int[2, n-S]
    edges = jnp.concatenate(
        (h_edges, v_edges),
        axis=1,
    ).transpose() # int[2(n-S), 2]

    # find random spanning tree
    include_edges = kruskal_brute(key, nodes, edges) # int[n-1, 2]
    # include_edges = kruskal_clever(key, nodes, edges) # int[n-1, 2]

    # build grid and carving out junctions/nodes
    grid = jnp.ones((size, size), dtype=bool)
    grid = grid.at[1::2,1::2].set(False)

    # carve out edges
    def edge_pos(edge_pair: Int[Array, "2"]) -> Int[Array, "2"]:
        divs, mods = jnp.divmod(edge_pair, S)
        rows = 1 + 2 * divs
        cols = 1 + 2 * mods
        row = jnp.sum(rows) // 2
        col = jnp.sum(cols) // 2
        return jnp.array((row, col))
    edge_poss = jax.vmap(edge_pos)(include_edges)
    grid = grid.at[
        edge_poss[:,0],
        edge_poss[:,1],
    ].set(False)

    return grid


def kruskal_clever(
    key: PRNGKeyArray,
    nodes: Int[Array, "n"],
    edges: Int[Array, "m 2"],
) -> Int[Array, "n-1 2"]:
    initial_parents = nodes
    
    edges = jax.random.permutation(
        key=key,
        x=edges,
        axis=0,
    )

    def _find(x, parents):
        px = parents[x]
        def _find_body_fn(args):
            x, px, parents = args
            ppx = parents[px]
            return px, ppx, parents.at[x].set(ppx)
        root, _, parents = jax.lax.while_loop(
            lambda args: args[0] != args[1],
            _find_body_fn,
            (x, px, parents),
        )
        return root, parents

    def try_edge(parents, edge):
        u, v = edge
        ru, parents = _find(u, parents)
        rv, parents = _find(v, parents)
        include_edge = (ru != rv)
        parents = parents.at[ru].set(rv)
        # ^this is a no-op if include_edge is false!
        return parents, include_edge

    _final_parents, include_edge_mask = jax.lax.scan(
        try_edge,
        initial_parents,
        edges,
    )
    # include_edge_mask : bool[m]
        
    include_edges = edges[
        jnp.where(include_edge_mask, size=nodes.size-1)
    ]
    return include_edges


def kruskal_brute(
    key: PRNGKeyArray,
    nodes: Int[Array, "n"],
    edges: Int[Array, "m 2"],
) -> Int[Array, "n-1 2"]:
    initial_parents = nodes
    
    edges = jax.random.permutation(
        key=key,
        x=edges,
        axis=0,
    )
    
    def try_edge(parents, edge):
        u, v = edge
        pu = parents[u]
        pv = parents[v]
        include_edge = (pu != pv)
        parents = jnp.where(
            include_edge,
            jnp.where(
                parents == pu,
                pv,
                parents,
            ),
            parents,
        )
        return parents, include_edge

    _final_parents, include_edge_mask = jax.lax.scan(
        try_edge,
        initial_parents,
        edges,
    )
    

    include_edges = edges[
        jnp.where(include_edge_mask, size=nodes.size-1)
    ]
    return include_edges

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
