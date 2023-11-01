import jax
import jax.numpy as jnp
# import dm_pix

def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2,), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2,), dtype=jnp.int32)])
    padded_img = jnp.pad(
        img, ((padding, padding), (padding, padding), (0, 0), (0, 0)), mode="edge"
    )
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)

# color jitter is currently not used
def random_crop_jitter(key, img, padding):
    rngs = jax.random.split(key, 7)
    img = random_crop(rngs[0], img, padding=4)
    img = (img.squeeze() / 255.).astype(jnp.float32)
    # Color jitter
    img_jt = img
    img_jt = img_jt * jax.random.uniform(rngs[1], shape=(1,), minval=0.5, maxval=1.5)  # Brightness
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_contrast(rngs[2], img_jt, lower=0.5, upper=1.5)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_saturation(rngs[3], img_jt, lower=0.5, upper=1.5)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    img_jt = dm_pix.random_hue(rngs[4], img_jt, max_delta=0.1)
    img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
    should_jt = jax.random.bernoulli(rngs[5], p=0.8)
    img = jnp.where(should_jt, img_jt, img)
    # Random grayscale
    # should_gs = jax.random.bernoulli(rngs[6], p=0.2)
    # img = jax.lax.cond(should_gs,  # Only apply grayscale if true
    #                    lambda x: dm_pix.rgb_to_grayscale(x, keep_dims=True),
    #                    lambda x: x,
    #                    img)
    # Gaussian blur
    # sigma = jax.random.uniform(rngs[7], shape=(1,), minval=0.1, maxval=2.0)
    # img = dm_pix.gaussian_blur(img, sigma=sigma[0], kernel_size=9)
    # Normalization
    # img = img * 2.0 - 1.0
    img = (img * 255.).astype(jnp.uint8)
    return img[..., None]

def batched_random_crop(key, obs, pixel_key, padding=4):
    imgs = obs[pixel_key]
    keys = jax.random.split(key, imgs.shape[0])
    imgs = jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)
    return obs.copy(add_or_replace={pixel_key: imgs})
