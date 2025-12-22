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
# Maze generator


@functools.partial(jax.jit, static_argnames=('size',))
def generate_maze(
    key: PRNGKeyArray,
    size: int, # >= 3, odd
) -> Bool[Array, "size size"]:
    """
    Generate a `size` by `size` binary gridworld with a 1-wall thick border and
    a random acyclic maze in the centre.

    Consider the 'junction' squares
        
        (1,1), (1,3), ..., (1,w-1), (3,1), ..., (size-1,size-1).
    
    These squares form the nodes of a grid graph. This function constructs a
    random spanning tree of this grid graph using Kruskal's algorithm, and
    returns the corresponding binary matrix.
    """
    # validate dimensions (static)
    assert size >= 3, "size must be at least 3"
    assert size % 2 == 1, "size must be odd"

    # assign each 'junction' in the grid an integer node id
    H, W = size // 2, size // 2
    nodes = jnp.arange(H * W)
    ngrid = nodes.reshape((H, W))

    # an edge between each pair of nodes (represented as a node id pair)
    # note: there are (H-1)W + H(W-1) = 2HW - H - W edges
    h_edges = jnp.stack((ngrid[:,:-1].flatten(), ngrid[:,1:].flatten()))
    v_edges = jnp.stack((ngrid[:-1,:].flatten(), ngrid[1:,:].flatten()))
    edges = jnp.concatenate((h_edges, v_edges), axis=-1).transpose()

    # kruskal's random spanning tree algorithm
    # include_edges = kruskal_clever(key, nodes, edges)
    include_edges = kruskal_brute(key, nodes, edges)

    # carve out junctions
    grid = jnp.ones((size, size), dtype=bool)
    grid = grid.at[1::2,1::2].set(False)
    
    # carve out edges
    def edge_pos(edge_ij: Int[Array, "2"]) -> Int[Array, "2"]:
        divs, mods = jnp.divmod(edge_ij, W)
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
    # initially each node is in its own subtree
    initial_parents = nodes

    # randomly shuffling the edges creates a random spanning tree
    edges = jax.random.permutation(key, edges, axis=0)

    # for each edge we decide whether to include it or skip it; tracking
    # connected subtrees with a sophisticated union-find data structure.

    def _find(x, parent):
        """
        Finds the root of x, while updating parents so that parent[i]
        points one step closer to the root of i for next time.
        """
        px = parent[x]
        def _find_body_fun(args):
            x, px, parents = args
            ppx = parents[px]
            return px, ppx, parents.at[x].set(ppx)
        root, _, parent = jax.lax.while_loop(
            lambda args: args[0] != args[1],
            _find_body_fun,
            (x, px, parent),
        )
        return root, parent

    def _union(root_x, root_y, parents):
        """
        Updates the root of x to be the root of y.
        """
        return parents.at[root_x].set(root_y)

    def try_edge(parents, edge):
        u, v = edge
        ru, parents = _find(u, parents)
        rv, parents = _find(v, parents)
        include_edge = (ru != rv)
        parents = jax.lax.cond(
            include_edge,
            _union,
            lambda rx, ry, ps: ps,
            ru,
            rv,
            parents,
        )
        return parents, include_edge

    _final_parents, include_edge_mask = jax.lax.scan(
        try_edge,
        initial_parents,
        edges,
    )

    # extract the pairs corresponding to the `n-1` included edges
    include_edges = edges[
        jnp.where(include_edge_mask, size=(nodes.size-1))
    ]
    return include_edges


def kruskal_brute(
    key: PRNGKeyArray,
    nodes: Int[Array, "n"],
    edges: Int[Array, "m 2"],
) -> Int[Array, "n-1 2"]:
    # initially each node is in its own subtree
    initial_parents = nodes

    # randomly shuffling the edges creates a random spanning tree
    edges = jax.random.permutation(key, edges, axis=0)

    # for each edge we decide whether to include or skip it;
    # track connected subtrees with a simple union-find data structure
    def try_edge(parents, edge):
        u, v = edge
        pu = parents[u]
        pv = parents[v]
        include_edge = (pu != pv)
        new_parents = jax.lax.select(
            include_edge & (parents == pv),
            jnp.full_like(parents, pu),
            parents,
        )
        return new_parents, include_edge
    
    _final_parents, include_edge_mask = jax.lax.scan(
        try_edge,
        initial_parents,
        edges,
    )

    # extract the pairs corresponding to the `n-1` included edges
    include_edges = edges[
        jnp.where(include_edge_mask, size=(nodes.size-1))
    ]
    return include_edges


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
        key_walls, key = jax.random.split(key)
        walls = generate_maze(
            key_walls,
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
        div, mod = jnp.divmod(indices, size//2)
        hero_pos = 1 + 2*jnp.array([div[0], mod[0]])
        goal_pos = 1 + 2*jnp.array([div[1], mod[1]])

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
        # check collisions
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
        # pick a random valid action
        probes = state.hero_pos + jnp.array([
            ( 0,  0),
            (-1,  0),
            ( 0, -1),
            (+1,  0),
            ( 0, +1),
        ])
        invalid = state.walls[probes[:,0], probes[:,1]].at[0].set(True)
        key_step, key = jax.random.split(key)
        action = jax.random.choice(
            key=key_step,
            a=len(Action),
            p=~invalid,
            shape=(),
        )
        # take a step
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
    size: int = 25,
    seed: int = 42,
):
    key = jax.random.key(seed=seed)

    print("initialise environment...")
    key_init, key = jax.random.split(key)
    state = Environment.init(key=key_init, size=size)
    print(mp.image(state.render()))


    print("scanned and vmapped random walk...")
    statess = jax.vmap(walk, in_axes=(0,None,None))(
        jax.random.split(key, 64),
        size,
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
        H=8,
        W=8,
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
        "../gallery/lecture10.gif",
        format="gif",
        save_all=True,
        append_images=[Image.fromarray(f) for f in frames[1:]],
        duration=20,
        loop=0,
    )


if __name__ == "__main__":
    tyro.cli(main)
