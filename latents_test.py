
import torch

def prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height // 2, width // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )
    # NEW: Set the condition area on the first channel
    latent_image_ids[:, :width//4, 0] = 1
    
    (
        latent_image_id_height,
        latent_image_id_width,
        latent_image_id_channels,
    ) = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width,
        latent_image_id_channels,
    )

    return latent_image_ids.to(device=device, dtype=dtype)


def prepare_latents(
    vae_scale_factor,
    batch_size,
    height,
    width,
    dtype,
    device,
):
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))


    latent_image_ids = prepare_latent_image_ids(
        batch_size, height, width, device, dtype
    )
    return latent_image_ids


prepare_latents(8, 1, 768, 1024, torch.float16, "cpu")