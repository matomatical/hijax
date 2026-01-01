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
import tyro
import tqdm
import matthewplotlib as mp

from typing import Self
from jaxtyping import Array, Float, Int, UInt8 as Byte, PRNGKeyArray, PyTree

import numpy as np
import jax
import jax.numpy as jnp
import einops


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
    num_steps_per_reset: int = 32,
    seed: int = 221,
):
    key = jax.random.key(seed=seed)

    
    print("loading byte corpus...")
    with open("../data/sherlock.txt") as file:
        data = str_to_array(file.read())
    print("  num tokens:", len(data))
    

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
    print(
        "  number of parameters:",
        sum(l.size for l in jax.tree.leaves(model)),
    )

    # print("testing model completion...")
    prompt = "  Sherlock Holmes and Doctor W"
    original_prompt_tokens = str_to_array(prompt)
    key_completion, key = jax.random.split(key)
    # completion_tokens = model.complete(
    #     key=key_completion,
    #     prompt_tokens=original_prompt_tokens,
    #     num_tokens_out=completion_length,
    # )
    # print(vis_example(
    #     prompt=prompt,
    #     completion=array_to_str(completion_tokens),
    #     t=0,
    #     T=num_steps,
    # ))

    
    print("initialising optimiser...")
    opt_state = Adam.init(
        model=model,
        alpha=learning_rate,
    )


    print("training loop...")
    prompt_tokens = original_prompt_tokens
    for step in tqdm.trange(num_steps):
        # sample a batch of sequences
        key_batch, key = jax.random.split(key)
        batch_ids = jax.random.choice(
            key=key_batch,
            a=data.size - max_context_length,
            shape=(batch_size, 1),
        ) + jnp.arange(max_context_length+1)
        data_batch = data[batch_ids]
        
        # compute loss, grad, update
        train_loss, grads = jax.value_and_grad(loss_fn)(
            model,
            data_batch,
        )
        delta, opt_state = opt_state.update(grads)
        model = jax.tree.map(jnp.add, model, delta)

        # show some completions!
        key_completion, key = jax.random.split(key)
        completion_tokens = model.complete(
            key=key_completion,
            prompt_tokens=prompt_tokens,
            num_tokens_out=completion_length // num_steps_per_reset,
        )
        plot = vis_example(
            prompt=array_to_str(prompt_tokens),
            completion=array_to_str(completion_tokens),
            t=step,
            T=num_steps,
        )
        if step % num_steps_per_reset == 0:
            tqdm.tqdm.write(str(plot))
        else:
            tqdm.tqdm.write(f"{-plot}{plot}")
        if step % num_steps_per_reset == num_steps_per_reset - 1:
            prompt_tokens = original_prompt_tokens
        else:
            prompt_tokens = jnp.concatenate((
                prompt_tokens,
                completion_tokens,
            ))
 
 
# # # 
# Helper functions


def str_to_array(s: str) -> Byte[Array, "len(s)"]:
    return jnp.array([ord(c) for c in s], dtype=jnp.uint8)


def array_to_str(a: Byte[Array, "len(s)"]) -> str:
    return "".join(chr(i) for i in a)


# # # 
# Architecture


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class AffineTransform:
    weights: Float[Array, "n_in n_out"]
    biases: Float[Array, "n_out"]

    @staticmethod
    def init(key: PRNGKeyArray, num_inputs: int, num_outputs: int) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return AffineTransform(
            weights=jax.random.uniform(
                key=key,
                shape=(num_inputs, num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
            biases=jnp.zeros(num_outputs),
        )

    def forward(
        w: Self,
        x: Float[Array, "n_in"],
    ) -> Float[Array, "n_out"]:
        return x @ w.weights + w.biases


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LinearTransform:
    weights: Float[Array, "n_in n_out"]

    @property
    def num_inputs(self) -> int:
        return self.weights.shape[0]

    @staticmethod
    def init(key: PRNGKeyArray, num_inputs: int, num_outputs: int) -> Self:
        bound = jax.lax.rsqrt(jnp.float32(num_inputs))
        return LinearTransform(
            weights=jax.random.uniform(
                key=key,
                shape=(num_inputs, num_outputs),
                minval=-bound,
                maxval=+bound,
            ),
        )

    def forward(
        w: Self,
        x: Float[Array, "n_in"],
    ) -> Float[Array, "n_out"]:
        return x @ w.weights


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class LayerNorm:
    loc: Float[Array, "size"]
    scale: Float[Array, "size"]

    def forward(
        self: Self,
        x: Float[Array, "size"],
    ) -> Float[Array, "size"]:
        x_mean = jnp.mean(x)
        x_rstd = jax.lax.rsqrt(jnp.var(x) + 1e-5)
        x_norm = (x - x_mean) * x_rstd
        return x_norm * self.scale + self.loc

    @staticmethod
    def init(
        key: PRNGKeyArray,
        size: int,
    ) -> Self:
        return LayerNorm(
            loc=jnp.zeros(size),
            scale=jnp.ones(size),
        )


@functools.partial(
    jax.tree_util.register_dataclass,
    meta_fields=["num_heads"],
    data_fields=["QKV", "output_transform"],
)
@dataclasses.dataclass
class MultiHeadedCausalSelfAttention:
    QKV: LinearTransform # LinearTransform[3]
    output_transform: LinearTransform
    num_heads: int

    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # perform query key and value transform
        forward = type(self.QKV).forward    # ., c -> c
        vforward = jax.vmap(
            forward,
            in_axes=(None, 0),
        )                                   # ., t c -> t c
        vvforward = jax.vmap(
            vforward,
            in_axes=(0, None),
        )                                   # qkv, t c -> qkv t c
        qkv = vvforward(self.QKV, x)

        # reshaping the embed dimension into separate heads
        qkv_perhead = einops.rearrange(
            qkv,
            'qkv t (num_heads head_size) -> qkv t num_heads head_size',
            num_heads=self.num_heads,
        )

        # vmap the attention computation across heads
        def single_head_attention(
            qkv: Float[Array, "3 t head_size"],
        ) -> Float[Array, "t head_size"]:
            q, k, v = qkv
            t, head_size = q.shape
            # compute raw affinities    tq c @ c tk -> tq tk
            a = (q @ k.T)
            # scale                     tq tk / . . -> tq tk
            a = a * jax.lax.rsqrt(jnp.float32(head_size))
            # causal mask
            a = jnp.where(
                jnp.tril(jnp.ones((t, t), dtype=bool)),
                a,
                -jnp.inf,
            )
            # converting affinities to mixing weights
            #                           tq tk -> tq prob(tk)
            p = jax.nn.softmax(a, axis=-1)
            # mix values for each key   tq prob(tk) @ tv c -> t c
            y = p @ v
            return y
        y_perhead = jax.vmap(
            single_head_attention,
            in_axes=2,
            out_axes=1,
        )(qkv_perhead) # -> t num_heads head_size

        # recombine heads into a new embed_size
        y = einops.rearrange(
            y_perhead,
            't num_heads head_size -> t (num_heads head_size)',
        )

        # output transform
        y_projected = jax.vmap(self.output_transform.forward)(y)

        return y_projected

    @staticmethod
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
    ) -> Self:
        k1, k2 = jax.random.split(key)
        return MultiHeadedCausalSelfAttention(
            QKV=jax.vmap(
                LinearTransform.init,
                in_axes=(0,None,None),
            )(
                jax.random.split(k1, 3),
                embed_size,
                embed_size,
            ),
            output_transform=LinearTransform.init(
                key=k2,
                num_inputs=embed_size,
                num_outputs=embed_size,
            ),
            num_heads=num_heads,
        )
    

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class MLP:
    layer1: AffineTransform # num_inputs -> num_hidden
    layer2: AffineTransform # num_hidden -> num_outputs

    def forward(
        self: Self,
        x: Float[Array, "num_inputs"],
    ) -> Float[Array, "num_outputs"]:
        x = self.layer1.forward(x)
        x = jax.nn.relu(x)
        x = self.layer2.forward(x)
        return x

    @staticmethod
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
    ) -> Self:
        k1, k2 = jax.random.split(key)
        return MLP(
            layer1=AffineTransform.init(
                key=k1,
                num_inputs=num_inputs,
                num_outputs=num_hidden,
            ),
            layer2=AffineTransform.init(
                key=k2,
                num_inputs=num_hidden,
                num_outputs=num_outputs,
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DecodeTransformerBlock:
    layernorm1: LayerNorm
    attention: MultiHeadedCausalSelfAttention
    layernorm2: LayerNorm
    compute: MLP

    def forward(
        self: Self,
        x: Float[Array, "t embed_size"],
    ) -> Float[Array, "t embed_size"]:
        # residual attention
        x_norm = jax.vmap(self.layernorm1.forward)(x)
        x = x + self.attention.forward(x_norm)
        # residual MLP
        x_norm = jax.vmap(self.layernorm2.forward)(x)
        x = x + jax.vmap(self.compute.forward)(x_norm)
        return x

    @staticmethod
    def init(
        key: PRNGKeyArray,
        embed_size: int,
        num_heads: int,
        mlp_size: int,
    ) -> Self:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return DecodeTransformerBlock(
            layernorm1=LayerNorm.init(
                key=k1,
                size=embed_size,
            ),
            attention=MultiHeadedCausalSelfAttention.init(
                key=k2,
                embed_size=embed_size,
                num_heads=num_heads,
            ),
            layernorm2=LayerNorm.init(
                key=k3,
                size=embed_size,
            ),
            compute=MLP.init(
                key=k4,
                num_inputs=embed_size,
                num_hidden=mlp_size,
                num_outputs=embed_size,
            ),
        )
        

@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DecodeTransformer:
    token_embedding: LinearTransform
    postn_embedding: LinearTransform
    blocks: DecodeTransformerBlock # [num_blocks]
    unembedding_layernorm: LayerNorm
    unembedding: AffineTransform

    @property
    def max_context_length(self) -> int:
        return self.postn_embedding.num_inputs

    def forward(
        self: Self,
        ts: Float[Array, "t num_inputs"],
    ) -> Float[Array, "t num_outputs"]:
        context_length, _num_inputs = ts.shape

        # embedding
        x_semantic = jax.vmap(self.token_embedding.forward)(ts)
        x_position = self.postn_embedding.weights[:context_length, :]
        x = x_semantic + x_position # t embed_size

        # apply the blocks
        x, _ = jax.lax.scan(
            lambda x, block: (block.forward(x), None),
            x,
            self.blocks,
        )

        # unembedding
        x_norm = jax.vmap(self.unembedding_layernorm.forward)(x)
        logits = jax.vmap(self.unembedding.forward)(x_norm)
        return logits

    @staticmethod
    def init(
        key: PRNGKeyArray,
        num_inputs: int,
        num_outputs: int,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ) -> Self:
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        return DecodeTransformer(
            token_embedding=LinearTransform.init(
                key=k1,
                num_inputs=num_inputs,
                num_outputs=embed_size,
            ),
            postn_embedding=LinearTransform.init(
                key=k2,
                num_inputs=max_context_length,
                num_outputs=embed_size,
            ),
            blocks=jax.vmap(
                DecodeTransformerBlock.init,
                in_axes=(0,None,None,None),
            )(
                jax.random.split(k3, num_blocks),
                embed_size,
                num_heads,
                mlp_size,
            ),
            unembedding_layernorm=LayerNorm.init(
                key=k4,
                size=embed_size,
            ),
            unembedding=AffineTransform.init(
                key=k5,
                num_inputs=embed_size,
                num_outputs=num_outputs,
            ),
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class ByteSequenceModel:
    decode_transformer: DecodeTransformer

    @property
    def max_context_length(self) -> int:
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
        byte_arrays: Byte[Array, "batch_size t"],
    ) -> Float[Array, "batch_size t 128"]:
        return jax.vmap(self.forward)(byte_arrays)

    @functools.partial(
        jax.jit,
        static_argnames=["num_tokens_out"],
    )
    def complete(
        self: Self,
        key: PRNGKeyArray,
        prompt_tokens: Byte[Array, "t"],
        num_tokens_out: int,
        inverse_temperature: float = 1.,
    ) -> Byte[Array, "num_tokens_out"]:
        num_tokens_in, = prompt_tokens.shape
        
        # initialise buffer
        padding = max(num_tokens_out, self.max_context_length-num_tokens_in)
        buffer = jnp.concatenate([
            prompt_tokens,
            jnp.zeros(padding, dtype=jnp.uint8),
        ])
        
        # # loop across the buffer
        # for i in range(num_tokens_out):
        #     # slice
        #     lo = max(0, num_tokens_in + i - self.max_context_length)
        #     hi = num_tokens_in + i
        #     window = buffer[lo:hi]
        #     # predict next token
        #     probs_next_token = self.forward(window)[-1]
        #     key_generate, key = jax.random.split(key)
        #     next_token = jax.random.choice(
        #         key=key_generate,
        #         a=128,
        #         p=probs_next_token ** inverse_temperature,
        #         shape=(),
        #     ).astype(dtype=jnp.uint8)
        #     # add token to buffer
        #     buffer = buffer.at[num_tokens_in+i].set(next_token)
        
        def step(i: int, carry):
            buffer, key = carry
            # slice
            lo = jnp.maximum(0, num_tokens_in + i - self.max_context_length)
            length = self.max_context_length
            window = jax.lax.dynamic_slice(
                operand=buffer,
                start_indices=(lo,),
                slice_sizes=(length,),
            )
            cursor = num_tokens_in + i - lo - 1
            # jax.debug.print("{} {} {}", i, window, cursor)
            # predict next token
            probs_next_token = self.forward(window)[cursor]
            key_generate, key = jax.random.split(key)
            next_token = jax.random.choice(
                key=key_generate,
                a=128,
                p=probs_next_token ** inverse_temperature,
                shape=(),
            ).astype(dtype=jnp.uint8)
            # add token to buffer
            buffer = buffer.at[num_tokens_in+i].set(next_token)
            return (buffer, key)
        buffer, key = jax.lax.fori_loop(
            lower=0,
            upper=num_tokens_out,
            body_fun=step,
            init_val=(buffer, key),
        )

        # return completion
        tokens_out = buffer[num_tokens_in:num_tokens_in+num_tokens_out]
        return tokens_out

    @staticmethod
    @functools.partial(
        jax.jit,
        static_argnames=[
            "max_context_length",
            "embed_size",
            "mlp_size",
            "num_heads",
            "num_blocks",
        ],
    )
    def init(
        key: PRNGKeyArray,
        max_context_length: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_blocks: int,
    ) -> Self:
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


# # # 
# Cross entropy functions


def cross_entropy(
    true_index: Int[Array, ""],
    pred_distr: Float[Array, "v"],
) -> float:
    return -jnp.log(pred_distr[true_index])


def loss_fn(
    model: ByteSequenceModel,
    tokens: Byte[Array, "b t+1"],
) -> float:
    true_indices = tokens[:,1:]                         # int[b,t]
    pred_distrs = model.batch_forward(tokens[:,:-1])    # float[b,t,v]
    cross_entropies = jax.vmap(jax.vmap(cross_entropy))(
        true_indices,
        pred_distrs,
    )
    return jnp.mean(cross_entropies)


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
# Visualisation


def wrap(text: str, max_width: int, max_lines: int) -> str:
    num_lines = min(len(text) // max_width + 1, max_lines)
    lines = [
        text[i:i+max_width]
        for i in range(0, num_lines*max_width, max_width)
    ]
    return "\n".join(lines)


def vis_example(
    prompt: str,
    completion: str,
    t: int,
    T: int,
) -> mp.plot:
    # strip quotes

    # prompt
    render_prompt = repr(prompt)[1:-1]
    wrapped_prompt = wrap(render_prompt, max_width=29, max_lines=13)
    plot_prompt = mp.text(wrapped_prompt, fgcolor="cyan")
    
    # completion
    render_completion = repr(completion)[1:-1]
    offset_completion = " "*len(render_prompt) + render_completion
    wrapped_completion = wrap(offset_completion, max_width=29, max_lines=13)
    plot_completion = mp.text(wrapped_completion, fgcolor='magenta')

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
