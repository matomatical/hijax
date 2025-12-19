"""
Lecture 08: Hi, static arguments!

Demonstration: Implement and train a byte transformer on the Sherlock Holmes
canon.

Learning objectives:

* further exploration of jax.jit (jit dojo lessons 4 and 5):
  * static arguments and valid types for static arguments
  * static fields in data classes
* dynamic slicing
"""

import dataclasses
import functools
import jax
import jax.numpy as jnp
import einops
import textwrap
import tyro
import tqdm
import numpy as np
import matthewplotlib as mp

from typing import Self
from jaxtyping import Array, Float, Int, UInt8 as Byte, PRNGKeyArray, PyTree


def main(
    # model config
    num_blocks: int = 6,
    num_heads: int = 8,
    embed_size: int = 256,
    mlp_size: int = 256,
    max_context_length: int = 64,
    completion_length: int = 256,
    # training config
    learning_rate: float = 0.0001,
    batch_size: int = 32,
    num_steps: int = 2048,
    num_steps_per_vis: int = 16,
    seed: int = 221,
):
    key = jax.random.key(seed=seed)

    
    print("loading byte corpus...")
    with open("sherlock.txt") as file:
        data = str_to_array(file.read())
    print("  number of training tokens (bytes):", *data.shape)
    

    print("configuring model architecture...")
    key_model, key = jax.random.split(key)
    model = ByteSequenceModel.init(
        key=key_model,
        max_context_length=max_context_length,
        embed_size=embed_size,
        mlp_size=mlp_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
    )
    print("parameters:", sum(jnp.size(x) for x in jax.tree.leaves(model)))


    print("testing model completion...")
    prompt = "  Sherlock Holmes and Doctor W"
    prompt_tokens = str_to_array(prompt)
    key_completion, key = jax.random.split(key)
    completion = model.complete(
        key=key,
        prompt_tokens=prompt_tokens,
        num_tokens_out=completion_length,
    )
    print(vis_example(
        prompt=prompt,
        completion=array_to_str(completion),
        uncover=1.0,
        t=0,
        T=num_steps,
    ))

    
    print("initialising optimiser...")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
    )


    print("training loop...")
    plots = []
    for step in tqdm.trange(num_steps):
        # sample a batch of sequences
        key_batch, key = jax.random.split(key)
        batch_ids = jax.random.choice(
            key=key_batch,
            a=data.size - max_context_length,
            shape=(batch_size, 1),
        ) + jnp.arange(max_context_length+1)
        data_batch = data[batch_ids]
        
        # compute the batch loss and grad
        train_loss, grads = jax.value_and_grad(loss_fn)(
            model,
            data_batch,
        )
        # compute update, update optimiser, update model
        delta, opt_state = opt_state.update(grads)
        model = jax.tree.map(jnp.add, model, delta)
        
        # periodically compute example
        i = step % num_steps_per_vis
        if i == 0:
            key_completion, key = jax.random.split(key)
            completion = model.complete(
                key=key_completion,
                prompt_tokens=prompt_tokens,
                num_tokens_out=completion_length,
            )
        plot = vis_example(
            prompt=prompt,
            completion=array_to_str(completion),
            uncover=i / (num_steps_per_vis - 1),
            t=step // num_steps_per_vis * num_steps_per_vis,
            T=num_steps,
        )
        if step == 0:
            tqdm.tqdm.write(str(plot))
        else:
            tqdm.tqdm.write(f"\x1b[{plot.height}A{plot}")
        plots.append(plot)

    mp.save_animation(
        plots,
        "../gallery/lecture08.gif",
        bgcolor="black",
        fps=25,
    )

 
 
# # # 
# Architecture


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LinearTransform:
    weights: Float[Array, "num_inputs num_outputs"]
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=["num_inputs", "num_outputs"])
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
    ) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        weights = jax.random.uniform(
            key=key,
            shape=(num_inputs, num_outputs),
            minval=-bound,
            maxval=+bound,
        )
        return LinearTransform(weights=weights)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        return x @ self.weights


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
    meta_fields=("num_heads",),
    data_fields=("QKV", "output_transform"),
)
@dataclasses.dataclass
class MultiHeadedCausalSelfAttention:
    QKV: LinearTransform # LinearTransform[3]
    output_transform: LinearTransform
    num_heads: int

    @staticmethod
    @functools.partial(jax.jit, static_argnames=["embed_size", "num_heads"])
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
    ) -> Self:
        key_qkv, key = jax.random.split(key)
        QKV = jax.vmap(
            LinearTransform.init,
            in_axes=(0,None,None),
        )(
            jax.random.split(key_qkv, 3),
            embed_size,
            embed_size,
        )
        key_out, key = jax.random.split(key)
        output_transform = LinearTransform.init(
            key_out,
            embed_size,
            embed_size,
        )
        return MultiHeadedCausalSelfAttention(
            QKV=QKV,
            output_transform=output_transform,
            num_heads=num_heads,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # perform query, key, value transformations (on all heads at once)
        qkv = jax.vmap(
            type(self.QKV).forward, # two-argument version of self.QKV.forward
            in_axes=(0, None),
        )(self.QKV, x)

        # reshape the embed dimension into separate heads
        qkv_perhead = einops.rearrange(
            qkv,
            'qkv t (num_heads head_size) -> qkv t num_heads head_size',
            num_heads=self.num_heads,
        )

        # vmap the attention computation across each head
        def single_head_attention(
            qkv: Float[Array, "3 t head_size"],
        ) -> Float[Array, "t head_size"]:
            q, k, v = qkv
            t, head_size = q.shape
            # compute raw affinities                tq c @ c tk -> tq tk
            a = (q @ k.T)                                   
            # scale                                 tq tk / . . -> tq tk
            a = a * jax.lax.rsqrt(jnp.float32(head_size))
            # apply causal mask                     tq tk + t t -> tq tk
            a = jnp.where(
                jnp.tril(jnp.ones((t, t), dtype=bool)), # lower triangular mask
                a,
                -jnp.inf,
            )
            # convert affinities to mixing weights  tq tk -> tq prob(tk)
            p = jax.nn.softmax(a, axis=-1)
            # mix values for each key               tq prob(tk) @ tv c -> t c
            y = p @ v
            return y
        y_perhead = jax.vmap(
            single_head_attention,
            in_axes=2,  # qkv t vmap(num_heads) head_size
            out_axes=1, #     t vmap(num_heads) head_size
        )(qkv_perhead)
        
        # recombine heads into new embedding dimension
        y = einops.rearrange(
            y_perhead,
            't num_heads head_size -> t (num_heads head_size)',
        )

        # for each token, project back into residual stream
        y_ = jax.vmap(self.output_transform.forward)(y)
        
        return y_
    

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MLP:
    layer1: AffineTransform
    layer2: AffineTransform

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=["num_inputs", "num_hidden", "num_outputs"],
    )
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ) -> Self:
        k1, k2 = jax.random.split(key)
        layer1 = AffineTransform.init(k1, num_inputs, num_hidden)
        layer2 = AffineTransform.init(k2, num_hidden, num_outputs)
        return MLP(layer1=layer1, layer2=layer2)

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = self.layer1.forward(x)
        x = jax.nn.relu(x)
        x = self.layer2.forward(x)
        return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LayerNorm:
    loc: Float[Array, "size"]
    scale: Float[Array, "size"]
    
    @staticmethod
    @functools.partial(jax.jit, static_argnames=["size"])
    def init(
        key: PRNGKeyArray,
        size: int,
    ) -> Self:
        return LayerNorm(
            loc=jnp.zeros(size),
            scale=jnp.ones(size),
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "size"],
    ) -> Float[Array, "size"]:
        x_mean = jnp.mean(x)
        x_rstd = jax.lax.rsqrt(jnp.var(x) + 1e-5)
        x_norm = (x - x_mean) * x_rstd
        return x_norm * self.scale + self.loc


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DecodeTransformerBlock:
    layernorm1: LayerNorm
    attention: MultiHeadedCausalSelfAttention
    layernorm2: LayerNorm
    compute: MLP

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=["embed_size", "num_heads", "mlp_size"],
    )
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
        mlp_size: int,
    ) -> Self:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        layernorm1 = LayerNorm.init(key=k1, size=embed_size)
        attention = MultiHeadedCausalSelfAttention.init(
            key=k2,
            embed_size=embed_size,
            num_heads=num_heads,
        )
        layernorm2 = LayerNorm.init(key=k3, size=embed_size)
        compute = MLP.init(
            key=k4,
            num_inputs=embed_size,
            num_hidden=mlp_size,
            num_outputs=embed_size,
        )
        return DecodeTransformerBlock(
            layernorm1=layernorm1,
            attention=attention,
            layernorm2=layernorm2,
            compute=compute,
        )

    @jax.jit
    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # pre layer norm (per-token)
        x_norm = jax.vmap(self.layernorm1.forward)(x)
        # attention (between tokens, residual)
        x = x + self.attention.forward(x_norm)
        # pre layer norm (per-token)
        x_norm = jax.vmap(self.layernorm2.forward)(x)
        # compute (per-token, residual)
        x = x + jax.vmap(self.compute.forward)(x_norm)
        return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DecodeTransformer:
    token_embedding: LinearTransform
    postn_embedding: LinearTransform
    blocks: DecodeTransformerBlock # DecodeTransformerBlock[num_blocks]
    unembedding_layernorm: LayerNorm
    unembedding: AffineTransform

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=[
            "num_inputs",
            "max_context_length",
            "num_blocks",
            "num_heads",
            "embed_size",
            "mlp_size",
            "num_outputs",
        ],
    )
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        max_context_length: int,
        num_blocks: int,
        num_heads: int,
        embed_size: int,
        mlp_size: int,
        num_outputs: int,
    ) -> Self:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # embeddings
        token_embedding = LinearTransform.init(
            k1,
            num_inputs,
            embed_size,
        )
        postn_embedding = LinearTransform.init(
            k2,
            max_context_length,
            embed_size,
        )
        
        # transformer blocks
        blocks = jax.vmap(
            DecodeTransformerBlock.init,
            in_axes=(0,None,None,None),
        )(
            jax.random.split(k3, num_blocks),
            embed_size,
            num_heads,
            mlp_size,
        )

        # unembedding
        unembedding_layernorm = LayerNorm.init(
            k4,
            embed_size,
        )
        unembedding = AffineTransform.init(
            k5,
            embed_size,
            num_outputs,
        )
        return DecodeTransformer(
            token_embedding=token_embedding,
            postn_embedding=postn_embedding,
            blocks=blocks,
            unembedding_layernorm=unembedding_layernorm,
            unembedding=unembedding,
        )

    @property
    def max_context_length(self: Self) -> int:
        return self.postn_embedding.weights.shape[0]

    @jax.jit
    def forward(
        self: Self,
        ts: Float[Array, "t num_inputs"],
    ) -> Float[Array, "t num_outputs"]:
        context_length, _num_inputs = ts.shape

        # embedding: semantic and positional token embeddings
        x_semantic = jax.vmap(self.token_embedding.forward)(ts)
        x_position = self.postn_embedding.weights[:context_length, :]
        x = x_semantic + x_position                         # -> t embed_size
        # apply the num_blocks attention blocks in sequence
        x, _ = jax.lax.scan(
            lambda x, block: (block.forward(x), None),
            x,
            self.blocks,
        )                                                   # -> t embed_size
        # unembedding: transform back to predicted next token probs
        x_norm = jax.vmap(self.unembedding_layernorm.forward)(x)
        x = jax.vmap(self.unembedding.forward)(x_norm)      # -> t num_outputs
        return x


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ByteSequenceModel:
    decode_transformer: DecodeTransformer

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=(
            "max_context_length",
            "embed_size",
            "mlp_size",
            "num_heads",
            "num_blocks",
        ),
    )
    def init(
        key: PRNGKeyArray, 
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ):
        return ByteSequenceModel(
            decode_transformer=DecodeTransformer.init(
                key=key,
                num_inputs=128,
                num_outputs=128,
                max_context_length=max_context_length,
                embed_size=embed_size,
                mlp_size=mlp_size,
                num_heads=num_heads,
                num_blocks=num_blocks,
            ),
        )

    @property
    def max_context_length(self: Self) -> int:
        return self.decode_transformer.max_context_length

    @jax.jit
    def forward(
        self: Self,
        byte_array: Byte[Array, "t"],
    ) -> Float[Array, "t 128"]:
        tokens_one_hot = jax.nn.one_hot(byte_array, num_classes=128)
        next_tokens_logits = self.decode_transformer.forward(tokens_one_hot)
        next_tokens_probs = jax.nn.softmax(next_tokens_logits, axis=-1)
        return next_tokens_probs

    @jax.jit
    def batch_forward(
        self: Self,
        byte_arrays: Byte[Array, "batch t"],
    ) -> Float[Array, "batch t 128"]:
        return jax.vmap(self.forward)(byte_arrays)

    # TODO: JIT, dynamic slice
    def complete(
        self: Self,
        key: PRNGKeyArray,
        prompt_tokens: Byte[Array, "num_tokens_in"],
        num_tokens_out: int,
        inverse_temperature: float = 1.,
    ) -> Byte[Array, "num_tokens_out"]:
        num_tokens_in, = prompt_tokens.shape
        # set up buffer we will slide across
        buffer = jnp.concatenate((
            prompt_tokens,
            jnp.zeros(num_tokens_out, dtype=jnp.uint8),
        ))
        lprompt = len(prompt_tokens)
        # loop across buffer
        keys_next_token = jax.random.split(key, num_tokens_out)
        for i, k in zip(range(num_tokens_out), keys_next_token):
            # slice window
            lo = max(0, lprompt+i-self.max_context_length)
            window = buffer[lo:lprompt+i]
            # predict next token
            prob_next_token = self.forward(window)[-1]
            next_token = jax.random.choice(
                key=k,
                a=128,
                p=prob_next_token ** inverse_temperature,
                shape=(),
            ).astype(dtype=jnp.uint8)
            # add token to buffer
            buffer = buffer.at[lprompt+i].set(next_token)
        # return completion
        tokens_out = buffer[-num_tokens_out:]
        return tokens_out


# # # 
# Helper functions


def str_to_array(s: str) -> Byte[Array, "len(s)"]:
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def array_to_str(a: Byte[Array, "len(s)"]) -> str:
    return "".join(chr(i) for i in a)


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
# Cross entropy functions


@jax.jit
def cross_entropy_dirac(
    true_index: Int[Array, ""],
    pred_distr: Float[Array, "v"],
) -> float:
    return -jnp.log(pred_distr[true_index])


@jax.jit
def batch_cross_entropy_dirac(
    true_indexs: Int[Array, "..."],
    pred_distrs: Float[Array, "... v"],
) -> Float[Array, "..."]:
    batched = jnp.vectorize(cross_entropy_dirac, signature='(),(v)->()')
    return batched(true_indexs, pred_distrs)
    

# # # 
# Loss function


@jax.jit
def loss_fn(
    model: ByteSequenceModel,
    tokens: Byte[Array, "b t+1"],
) -> float:
    return batch_cross_entropy_dirac(
        true_indexs=tokens[:,1:],
        pred_distrs=model.batch_forward(tokens[:,:-1]),
    ).mean()


# # # 
# Visualisation


def vis_example(
    prompt: str,
    completion: str,
    uncover: float,
    t: int,
    T: int,
) -> mp.plot:
    # proportionally truncate completion
    truncated_completion = completion[:int(len(completion) * uncover)]

    # strip quotes
    render_prompt = repr(prompt)[1:-1]
    render_completion = repr(truncated_completion)[1:-1]

    wrapper = textwrap.TextWrapper(
        width=29,
        replace_whitespace=False,
        drop_whitespace=False,
        max_lines=13,
        placeholder="...",
    )

    # prompt
    plot_prompt = mp.text(
        '\n'.join(wrapper.wrap(render_prompt)),
        fgcolor="cyan",
    )
    
    # completion
    wrapped_completion = '\n'.join(wrapper.wrap(
        " "*len(render_prompt) + render_completion,
    ))
    plot_completion = mp.text(
        wrapped_completion,
        fgcolor='magenta',
    )

    return mp.border(
        mp.dstack(
            mp.blank(width=29, height=13),
            plot_completion,
            plot_prompt,
        ),
        title=f"completion at {t:4d}/{T:4d}",
    )


if __name__ == "__main__":
    tyro.cli(main)
