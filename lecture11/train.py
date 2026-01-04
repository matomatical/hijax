"""
Lecture 11: Hi, deep reinforcement learning!

Demonstration: Fully accelerated RL training loop, PPO with GAE. Train a policy
to solve a small maze.

Learning objectives:

* review of many topics discussed in previous lectures
* reverse scan
"""

import dataclasses
import enum
import functools
import tqdm
import tyro
import matthewplotlib as mp

from jaxtyping import Int, Bool, Float, Array, PRNGKeyArray, PyTree
from typing import Self, Callable

import jax
import jax.numpy as jnp
import einops


def main(
    # configure environment
    size: int = 13,
    num_parallel_envs: int = 32,
    num_steps_per_rollout: int = 64,
    # configure agent
    net_channels: int = 6,
    net_width: int = 32,
    num_conv_layers: int = 1,
    num_dense_layers: int = 1,
    # configure training
    num_steps: int = 256,
    learning_rate: float = 0.001,
    discount_rate: float = 0.995,
    eligibility_rate: float = 0.95,
    proximity_eps: float = 0.1,
    critic_coeff: float = 0.5,
    entropy_coeff: float = 0.001,
    # other
    num_tests: int = 16,
    vis_grid_width: int = 4,
    seed: int = 42,
):
    key = jax.random.key(seed=seed)

    print("initialise environment...")
    key_init_env, key = jax.random.split(key)
    state = Environment.init(
        key=key_init_env,
        size=size,
    )
    print(mp.image(state.observe().astype(float)))


    print("initialise agent...")
    key_init_model, key = jax.random.split(key)
    model = ActorCriticNetwork.init(
        key=key_init_model,
        size=size,
        net_channels=net_channels,
        net_width=net_width,
        num_conv_layers=num_conv_layers,
        num_dense_layers=num_dense_layers,
        num_actions=len(Action),
    )
    pi, v = model.forward(state.observe())
    print(" action probs:", jax.nn.softmax(pi))
    print(" state value: ", v)
    

    print("initialising optimiser...")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
    )


    print("initialise test environments...")
    key_tests, key = jax.random.split(key)
    tests = jax.vmap(state.reset)(jax.random.split(key_tests, num_tests))
    plot = vis_grid(tests, grid_width=vis_grid_width)
    print(plot)

    for step in tqdm.trange(num_steps):
        # training step
        key, model, opt_state, mean_reward = ppo_train_step(
            env_state=state,
            key=key,
            model=model,
            opt_state=opt_state,
            num_steps_per_rollout=num_steps_per_rollout,
            num_parallel_envs=num_parallel_envs,
            eligibility_rate=eligibility_rate,
            discount_rate=discount_rate,
            proximity_eps=proximity_eps,
            critic_coeff=critic_coeff,
            entropy_coeff=entropy_coeff,
        )
        # animation step
        key_vis, key = jax.random.split(key)
        tests = jax.vmap(animation_step, in_axes=(0,0,None))(
            jax.random.split(key_vis, num_tests),
            tests,
            model,
        )
        plot = vis_grid(tests, grid_width=vis_grid_width)
        tqdm.tqdm.write(f"{-plot}{plot}")

    print("done!")


# # # 
# Maze generator


@functools.partial(jax.jit, static_argnames=('size',))
def generate_maze(
    key: PRNGKeyArray,
    size: int, # >= 3, odd
) -> Bool[Array, "size size"]:
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
    include_edges = kruskal(key, nodes, edges)

    # finally, generate the grid array
    grid = jnp.ones((size, size), dtype=bool)
    grid = grid.at[1::2,1::2].set(False)        # carve out junctions
    include_edges_ijs = jnp.rint(jnp.stack((    # carve out edges
        include_edges // W,
        include_edges % W,
    )).mean(axis=-1) * 2 + 1).astype(int)
    grid = grid.at[tuple(include_edges_ijs)].set(False)

    return grid


def kruskal(
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
# Environment


class Action(enum.IntEnum):
    WAIT = 0 # do nothing
    UP = 1 # move up
    LEFT = 2 # move left
    DOWN = 3 # move down
    RIGHT = 4 # move right


type Observation = Bool[Array, "size size 3"]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Environment:
    hero_pos: Int[Array, "2"]
    goal_pos: Int[Array, "2"]
    walls: Bool[Array, "size size"]

    @property
    def size(self: Self) -> int:
        h, w = self.walls.shape
        return h

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

        return Environment(
            hero_pos=hero_pos,
            goal_pos=goal_pos,
            walls=walls,
        )

    @jax.jit
    def reset(self: Self, key: PRNGKeyArray) -> Self:
        # generate random hero position
        key_pos, key = jax.random.split(key)
        index = jax.random.choice(
            key=key_pos,
            a=(self.size//2)**2,
            shape=(1,),
            replace=False,
        )
        div, mod = jnp.divmod(index, self.size//2)
        hero_pos = 1 + 2*jnp.array([div[0], mod[0]])
        
        return dataclasses.replace(
            self,
            hero_pos=hero_pos,
        )

    @jax.jit
    def observe(self: Self) -> Observation:
        size = self.walls.shape[0]
        obs = jnp.zeros((size, size, 3), dtype=bool)
        obs = obs.at[:,:,0].set(self.walls)
        obs = obs.at[self.hero_pos[0], self.hero_pos[1], 1].set(True)
        obs = obs.at[self.goal_pos[0], self.goal_pos[1], 2].set(True)
        return obs

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
        
    
# # # 
# Architecture


type ActorCriticFunction = Callable[
    [Observation],
    tuple[
        Float[Array, "actions"],
        float,
    ],
]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AffineTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    biases: Float[Array, "num_outputs"]

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights=jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        biases=jnp.zeros(num_outputs)
        return AffineTransform(weights=weights, biases=biases)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights + self.biases


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=("stride_size", "pad_same"),
    data_fields=("kernel",),
)
@dataclasses.dataclass
class Convolution:
    kernel: Float[Array, "channels_out channels_in kernel_size kernel_size"]
    stride_size: int
    pad_same: bool

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "channels_in",
            "channels_out",
            "kernel_size",
            "stride_size",
            "pad_same",
        ),
    )
    def init(
        key: PRNGKeyArray,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        stride_size: int,
        pad_same: bool,
    ) -> Self:
        num_inputs = channels_in * kernel_size**2
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return Convolution(
            kernel=jax.random.uniform(
                key=key,
                shape=(channels_out, channels_in, kernel_size, kernel_size),
                minval=-bound,
                maxval=+bound,
            ),
            stride_size=stride_size,
            pad_same=pad_same,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "height_in width_in channels_in"],
    ) -> Float[Array, "height_out width_out channels_out"]:
        x_1hwc = einops.rearrange(x, 'h w c -> 1 h w c')
        y_1hwc = jax.lax.conv_general_dilated(
            lhs=x_1hwc,
            rhs=self.kernel,
            window_strides=(self.stride_size, self.stride_size),
            padding="SAME" if self.pad_same else "VALID",
            dimension_numbers=("NHWC", "OIHW", "NHWC"),
        )
        return y_1hwc[0]


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ActorCriticNetwork:
    conv0: Convolution
    convs: Convolution # ["num_conv_layers-1"]
    dense0: AffineTransform
    denses: AffineTransform # ["num_dense_layers-1"]
    actor_head: AffineTransform
    critic_head: AffineTransform

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "size",
            "net_channels",
            "net_width",
            "num_conv_layers",
            "num_dense_layers",
            "num_actions",
        ),
    )
    def init(
        key: PRNGKeyArray,
        size: int,
        net_channels: int,
        net_width: int,
        num_conv_layers: int,
        num_dense_layers: int,
        num_actions: int,
    ):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        # initialise convolutional layers
        conv0 = Convolution.init(
            key=k1,
            channels_in=3,
            channels_out=net_channels,
            kernel_size=3,
            stride_size=1,
            pad_same=True,
        )
        convs = jax.vmap(
            Convolution.init,
            in_axes=(0, None, None, None, None, None),
        )(
            jax.random.split(k2, num_conv_layers-1),
            net_channels,
            net_channels,
            3,
            1,
            True,
        )
        # initialise dense layers
        dense0 = AffineTransform.init(
            key=k3,
            num_inputs=size ** 2 * (net_channels + 3),
            num_outputs=net_width,
        )
        denses = jax.vmap(
            AffineTransform.init,
            in_axes=(0, None, None),
        )(
            jax.random.split(k4, num_dense_layers-1),
            net_width,
            net_width,
        )
        # initialise critic / actor heads
        actor_head = AffineTransform.init(
            key=k5,
            num_inputs=net_width,
            num_outputs=num_actions,
        )
        critic_head = AffineTransform.init(
            key=k6,
            num_inputs=net_width,
            num_outputs=1,
        )
        return ActorCriticNetwork(
            conv0=conv0,
            convs=convs,
            dense0=dense0,
            denses=denses,
            actor_head=actor_head,
            critic_head=critic_head,
        )

    @jax.jit
    def forward(
        self: Self,
        obs: Observation,
    ) -> tuple[
        Float[Array, "num_actions"],
        Float[Array, ""],
    ]:
        obs = obs.astype(float)
        # embed observation part with residual CNN
        x = self.conv0.forward(obs)
        x = jax.nn.relu(x)
        x, _ = jax.lax.scan(
            lambda x, conv: (x + jax.nn.relu(conv.forward(x)), None),
            x,
            self.convs,
        )
        # combine with flattened input
        x = jnp.concatenate((jnp.ravel(x), jnp.ravel(obs)))
        # apply residual dense network
        x = self.dense0.forward(x)
        x = jax.nn.relu(x)
        x, _ = jax.lax.scan(
            lambda x, dense: (x + jax.nn.relu(dense.forward(x)), None),
            x,
            self.denses,
        )
        # apply action/value heads
        action_logits = self.actor_head.forward(x)
        value_pred = self.critic_head.forward(x)[0]
        return action_logits, value_pred
    
    @jax.jit
    def policy(
        self: Self,
        obs: Observation,
    ) -> Float[Array, "num_actions"]:
        pi, _v = self.forward(obs)
        return pi


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
# Reward


@jax.jit
def reward_fn(
    state: Environment,
    action: Action,
    state_: Environment,
) -> float:
    return state_.goal().astype(float)


# # # 
# Collect experience


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Transition:
    state: Environment
    obs: Observation
    value_pred: float
    action_logits: Float[Array, "num_actions"]
    action: Action
    next_state: Environment


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class Rollout:
    transitions: Transition
    final_value_pred: float


@functools.partial(jax.jit, static_argnames=("length", "policy"))
def collect_rollout(
    key: PRNGKeyArray,
    state: Environment,
    policy: ActorCriticFunction,
    length: int,
) -> Rollout:
    # define the initial carry (key, state)
    key_reset, key = jax.random.split(key)
    initial_state = state.reset(key=key_reset)
    initial_carry = (key, initial_state)

    # define a step of the random walk
    def step(carry, input_):
        key, state = carry
        # transition
        # sample an action
        key_step, key = jax.random.split(key)
        obs = state.observe()
        action_logits, value_pred = policy(obs)
        action = jax.random.choice(
            key=key_step,
            a=len(Action),
            p=jax.nn.softmax(action_logits),
            shape=(),
        )
        # take a step
        next_state = state.step(action)
        transition = Transition(
            state=state,
            obs=obs,
            value_pred=value_pred,
            action_logits=action_logits,
            action=action,
            next_state=next_state,
        )

        # reset if hit goal
        key_reset, key = jax.random.split(key)
        reset_flag = next_state.goal()
        reset_state = next_state.reset(key=key_reset)
        carry_state = jax.tree.map(
            lambda old, new: jnp.where(reset_flag, new, old),
            next_state,
            reset_state,
        )
        # output
        return (key, carry_state), transition

    # scan the walk
    final_carry, transitions = jax.lax.scan(
        step,
        initial_carry,
        length=length,
    )
    key, final_state = final_carry
    _pi, final_value_pred = policy(final_state.observe())
    return Rollout(
        transitions=transitions,
        final_value_pred=final_value_pred,
    )


# # # 
# Generalised advantage estimation


@jax.jit
def generalised_advantage_estimation(
    rewards: Float[Array, "num_steps"],
    values: Float[Array, "num_steps"],
    final_value: float,
    eligibility_rate: float,
    discount_rate: float,
) -> Float[Array, "num_steps"]:
    # reverse scan through num_steps axis
    initial_gae_and_next_value = (0., final_value)
    transitions = (rewards, values)
    def _gae_reverse_step(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        reward, this_value = transition
        gae = (
            reward
            - this_value
            + discount_rate * (next_value + eligibility_rate * gae)
        )
        return (gae, this_value), gae
    _final_carry, gaes = jax.lax.scan(
        _gae_reverse_step,
        initial_gae_and_next_value,
        transitions,
        reverse=True,
    )
    return gaes


# # # 
# PPO loss function


@jax.jit
def ppo_loss_fn(
    model: ActorCriticNetwork,
    transitions: Transition, # ["num_envs num_steps"]
    advantages: Float[Array, "num_envs num_steps"],
    discount_rate: float,
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> float:
    # reshape the data to have one batch dimension
    transitions, advantages = jax.tree.map(
        lambda x: einops.rearrange(
            x,
            "n_envs n_steps ... -> (n_envs n_steps) ...",
        ),
        (transitions, advantages),
    )
    batch_size = advantages.size

    # run network to get latest predictions
    new_action_logits, new_value_preds = jax.vmap(model.forward)(
        transitions.obs,
    ) # -> float[batch_size, 5], float[batch_size]

    # actor loss
    new_action_logprobs = jax.nn.log_softmax(
        new_action_logits,
        axis=1,
    )
    new_chosen_logprobs = new_action_logprobs[
        jnp.arange(batch_size),
        transitions.action,
    ]
    old_action_logprobs = jax.nn.log_softmax(
        transitions.action_logits,
        axis=1,
    )
    old_chosen_logprobs = old_action_logprobs[
        jnp.arange(batch_size),
        transitions.action,
    ]
    action_log_ratios = new_chosen_logprobs - old_chosen_logprobs
    action_prob_ratios = jnp.exp(action_log_ratios)
    action_prob_ratios_clipped = jnp.clip(
        action_prob_ratios,
        1 - proximity_eps,
        1 + proximity_eps,
    )
    std_advantages = (
        (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    )
    actor_loss = -jnp.minimum(
        std_advantages * action_prob_ratios,
        std_advantages * action_prob_ratios_clipped,
    ).mean()

    # critic loss
    value_diffs = new_value_preds - transitions.value_pred
    value_diffs_clipped = jnp.clip(
        value_diffs,
        -proximity_eps,
        proximity_eps,
    )
    new_value_preds_proximal = transitions.value_pred + value_diffs_clipped
    targets = transitions.value_pred + advantages
    critic_loss = jnp.maximum(
        jnp.square(new_value_preds - targets),
        jnp.square(new_value_preds_proximal - targets),
    ).mean() / 2

    # entropy regularisation term
    per_step_entropy = - jnp.sum(
        jnp.exp(new_action_logprobs) * new_action_logprobs,
        axis=1,
    )
    average_entropy = jnp.mean(per_step_entropy)

    # total loss
    total_loss = (
        actor_loss
        + critic_coeff * critic_loss
        - entropy_coeff * average_entropy
    )
    return total_loss


@functools.partial(
    jax.jit,
    static_argnames=(
        'num_steps_per_rollout',
        'num_parallel_envs',
    ),
)
def ppo_train_step(
    env_state: Environment,
    key: PRNGKeyArray,
    model: ActorCriticNetwork,
    opt_state: Adam,
    num_steps_per_rollout: int,
    num_parallel_envs: int,
    eligibility_rate: float,
    discount_rate: float,
    proximity_eps: float,
    critic_coeff: float,
    entropy_coeff: float,
) -> tuple[
    PRNGKeyArray,
    ActorCriticNetwork,
    Adam,
    float,
]:
    # collect experience with current policy...
    key_rollouts, key = jax.random.split(key)
    rollouts = jax.vmap(
        collect_rollout,
        in_axes=(0, None, None, None),
    )(
        jax.random.split(key_rollouts, num_parallel_envs),
        env_state,
        model.forward,
        num_steps_per_rollout,
    )
    # compute rewards
    rewards = jax.vmap(jax.vmap(reward_fn))(
        rollouts.transitions.state,
        rollouts.transitions.action,
        rollouts.transitions.next_state,
    )
    
    # estimate advantages on the collected experience...
    advantages = jax.vmap(
        generalised_advantage_estimation,
        in_axes=(0, 0, 0, None, None),
    )(
        rewards,
        rollouts.transitions.value_pred,
        rollouts.final_value_pred,
        eligibility_rate,
        discount_rate,
    )
    
    # update the policy on the collected experience...
    loss, grads = jax.value_and_grad(ppo_loss_fn)(
        model,
        transitions=rollouts.transitions,
        advantages=advantages,
        discount_rate=discount_rate,
        proximity_eps=proximity_eps,
        critic_coeff=critic_coeff,
        entropy_coeff=entropy_coeff,
    )
    delta, opt_state = opt_state.update(grads)
    model = jax.tree.map(jnp.add, model, delta)
    return (
        key,
        model,
        opt_state,
        rewards.mean(),
    )


# # # 
# Visualisation


def animation_step(
    key: PRNGKeyArray,
    state: Environment,
    model: ActorCriticNetwork,
) -> Environment:
    key_step, key = jax.random.split(key)
    action = jax.random.choice(
        key=key_step,
        a=len(Action),
        p=jax.nn.softmax(model.policy(state.observe())),
        shape=(),
    )
    next_state = state.step(action)
    reset_flag = next_state.goal()
    key_reset, key = jax.random.split(key)
    reset_state = next_state.reset(key=key_reset)
    next_state = jax.tree.map(
        lambda old, new: jnp.where(reset_flag, new, old),
        next_state,
        reset_state,
    )
    return next_state


def vis_grid(
    states: Environment,    # Environment[n]
    grid_width: int,        # grid_width divides n
) -> mp.plot:
    images = jax.vmap(Environment.render)(states)
    image = einops.rearrange(
        images,
        '(H W) h w rgb -> (H h) (W w) rgb',
        W=grid_width,
    )
    return mp.image(image)
    

# # # 
# Entry point


if __name__ == "__main__":
    tyro.cli(main)
