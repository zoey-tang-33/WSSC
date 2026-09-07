"""
WSSC: Wavelet-guided Semantic Signal Compensation for FLUX image editing.

Implements the WSSC inversion-free editing procedure for a FLUX
flow-matching pipeline, extended with:
  - a wavelet-domain "semantic signal compensation" that addresses the
    semantic indistinguishability problem: in the high-noise regime, the
    raw edit signal V_delta = V(zt_tar, tar) - V(zt_src, src) is dominated
    by the manifold-seeking flow, providing only weak and unreliable
    semantic directionality;
  - an optional external mask that restricts where the edit signal is allowed
    to propagate.

Public entry point: `WSSC`.
"""

from typing import Optional, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps


def scale_noise(
    scheduler,
    sample: torch.FloatTensor,
    timestep: Union[float, torch.FloatTensor],
    noise: Optional[torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Forward process in flow-matching.

    Args:
        sample (`torch.FloatTensor`): The input sample.
        timestep (`int`, *optional*): The current timestep in the diffusion chain.

    Returns:
        `torch.FloatTensor`: A scaled input sample.
    """
    scheduler._init_step_index(timestep)
    sigma = scheduler.sigmas[scheduler.step_index]
    sample = sigma * noise + (1.0 - sigma) * sample
    return sample


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def calc_v_flux(pipe, latents, prompt_embeds, pooled_prompt_embeds, guidance, text_ids, latent_image_ids, t):
    """Single FLUX transformer forward pass -> predicted velocity."""
    timestep = t.expand(latents.shape[0])

    with torch.no_grad():
        noise_pred = pipe.transformer(
            hidden_states=latents,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            pooled_projections=pooled_prompt_embeds,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

    return noise_pred


# ============================================================
# Haar 2D DWT / IDWT (pure PyTorch, GPU-native, zero deps)
# ============================================================

def haar_dwt2d(x):
    """
    Single-level 2D Haar wavelet transform.
    Input: x (B, C, H, W), H and W must be even.
    Output: cA (B, C, H/2, W/2), (cH, cV, cD) each (B, C, H/2, W/2)
      cA = approximation coefficients (low frequency)
      cH = horizontal detail (high frequency along the vertical direction)
      cV = vertical detail (high frequency along the horizontal direction)
      cD = diagonal detail (high frequency in both directions)
    """
    a = x[:, :, 0::2, 0::2]  
    b = x[:, :, 0::2, 1::2]  
    c = x[:, :, 1::2, 0::2]  
    d = x[:, :, 1::2, 1::2]  
    cA = (a + b + c + d) * 0.5
    cH = (a - b + c - d) * 0.5
    cV = (a + b - c - d) * 0.5
    cD = (a - b - c + d) * 0.5
    return cA, (cH, cV, cD)


def haar_idwt2d(cA, details):
    """
    Single-level 2D inverse Haar wavelet transform.
    Input: cA (B, C, H, W), details = (cH, cV, cD) each (B, C, H, W)
    Output: x (B, C, 2H, 2W)
    """
    cH, cV, cD = details
    B, C, H, W = cA.shape
    out = torch.zeros(B, C, H * 2, W * 2, device=cA.device, dtype=cA.dtype)
    out[:, :, 0::2, 0::2] = (cA + cH + cV + cD) * 0.5
    out[:, :, 0::2, 1::2] = (cA - cH + cV - cD) * 0.5
    out[:, :, 1::2, 0::2] = (cA + cH - cV - cD) * 0.5
    out[:, :, 1::2, 1::2] = (cA - cH - cV + cD) * 0.5
    return out


def haar_dwt2d_multilevel(x, n_levels=3):
    """
    Multi-level 2D Haar wavelet decomposition.
    Returns: (cA_n, [(details_n, orig_size_n), ...])
      details are ordered from the coarsest to the finest level.
    """
    coeffs_details = []
    current = x
    for _ in range(n_levels):
        # Pad to even size if the current size is odd.
        _, _, H, W = current.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            current = torch.nn.functional.pad(current, (0, pad_w, 0, pad_h), mode="reflect")
        cA, details = haar_dwt2d(current)
        coeffs_details.append((details, (H, W)))  # keep the original size for reconstruction
        current = cA
    return current, coeffs_details


def haar_idwt2d_multilevel(cA, coeffs_details):
    """
    Multi-level 2D Haar wavelet reconstruction.
    coeffs_details: [(details_n, orig_size_n), ..., (details_1, orig_size_1)]
      ordered from the coarsest to the finest level.
    """
    current = cA
    for details, (orig_H, orig_W) in reversed(coeffs_details):
        current = haar_idwt2d(current, details)
        current = current[:, :, :orig_H, :orig_W]  # crop the padding back off
    return current


def wavelet_extract_low_freq(
    V_packed,
    pipe,
    orig_height,
    orig_width,
    num_channels_latents,
    n_levels,
):
    """
    Decompose a packed-format velocity field with an n_levels Haar wavelet
    transform, keep only the coarsest-level approximation coefficients cA,
    zero out all detail coefficients, and reconstruct.
    Returns a packed-format signal containing only the low-frequency
    semantic component.
    """
    V_spatial = pipe._unpack_latents(
        V_packed, orig_height, orig_width, pipe.vae_scale_factor
    )
    B, C, H, W = V_spatial.shape
    dtype_orig = V_spatial.dtype
    V_spatial = V_spatial.float()

    actual_levels = min(n_levels, int(np.log2(max(min(H, W), 1))))
    actual_levels = max(actual_levels, 1)

    cA, coeffs_details = haar_dwt2d_multilevel(V_spatial, n_levels=actual_levels)

    # Zero out all detail coefficients (keep only the low-frequency approximation).
    zeroed_details = []
    for (cH, cV, cD), orig_size in coeffs_details:
        zeroed_details.append(
            ((torch.zeros_like(cH), torch.zeros_like(cV), torch.zeros_like(cD)), orig_size)
        )

    V_low = haar_idwt2d_multilevel(cA, zeroed_details)
    V_low = V_low[:, :, :H, :W].to(dtype_orig)

    V_low_packed = pipe._pack_latents(V_low, B, num_channels_latents, H, W)
    return V_low_packed


def expand_mask(mask, kernel_size=3, iters=1):
    """Dilate a binary/soft mask with repeated max-pooling."""
    for _ in range(iters):
        mask = F.max_pool2d(mask, kernel_size, stride=1, padding=kernel_size // 2)
    return mask


def WSSC(
    pipe,
    scheduler,
    x_src,
    src_prompt,
    tar_prompt,
    negative_prompt,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
    spectral_boost: float = 3.0,          # semantic compensation strength
    mask_path: Union[str, Path, None] = None,  # path to the mask
    use_mask: bool = True,
    n_levels: int = 3,
):
    """
    WSSC: Wavelet-guided semantic signal compensation for FLUX image editing.

    In the high-noise regime, the geometric edit signal
        V_delta = V(zt_tar, tar) - V(zt_src, src)
    is dominated by the manifold-seeking flow, providing only weak and
    unreliable semantic directionality (semantic indistinguishability).

    Fix: at every timestep, additionally compute a same-point semantic
    probe signal
        V_semantic = V(zt_src, tar) - V(zt_src, src)
    i.e. evaluate the tar- and src-conditioned velocities at the *same* point
    zt_src. Since both evaluations share the same input latent, the geometric
    confound is removed, yielding a cleaner semantic direction.

    Decompose V_semantic with a multi-level Haar wavelet transform to extract
    its low-frequency component (global semantics: color, pose), weight it by
    spectral_boost * t^2, and add it into the ODE as a compensation term:

        dZ = (V_delta + spectral_boost * t^2 * LowFreq(V_semantic)) * dt

    Effect:
      - t -> 0 (low noise): the compensation weight vanishes, recovering the
        baseline inversion-free edit exactly.
      - t -> 1 (high noise): the compensation term supplies a genuine semantic
        edit direction where V_delta is least informative.
      - spectral_boost = 0: equivalent to the baseline without compensation.

    Extra cost: +1 model forward pass per timestep (the semantic probe),
    outside the n_avg loop.

    When use_mask=True, an external mask (a grayscale image at mask_path) is
    also used to restrict the edit signal to the specified region.
    """

    device = x_src.device
    orig_height, orig_width = (
        x_src.shape[2] * pipe.vae_scale_factor // 2,
        x_src.shape[3] * pipe.vae_scale_factor // 2,
    )
    num_channels_latents = pipe.transformer.config.in_channels // 4

    pipe.check_inputs(
        prompt=src_prompt,
        prompt_2=None,
        height=orig_height,
        width=orig_width,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=512,
    )

    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_height,
        width=orig_width,
        dtype=x_src.dtype,
        device=x_src.device,
        generator=None,
        latents=x_src,
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3]
    )
    latent_tar_image_ids = latent_src_image_ids

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = retrieve_timesteps(
        scheduler,
        T_steps,
        device,
        timesteps=None,
        sigmas=sigmas,
        mu=mu,
    )
    pipe._num_timesteps = len(timesteps)

    # src prompt
    (
        src_prompt_embeds,
        src_pooled_prompt_embeds,
        src_text_ids,
    ) = pipe.encode_prompt(
        prompt=src_prompt,
        prompt_2=None,
        device=device,
    )

    # tar prompt
    pipe._guidance_scale = tar_guidance_scale
    (
        tar_prompt_embeds,
        tar_pooled_prompt_embeds,
        tar_text_ids,
    ) = pipe.encode_prompt(
        prompt=tar_prompt,
        prompt_2=None,
        device=device,
    )

    # handle guidance
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device)
        src_guidance = src_guidance.expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device)
        tar_guidance = tar_guidance.expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # initialize our ODE Zt_edit_1 = x_src
    zt_edit = x_src_packed.clone()
    

    if use_mask:
        if mask_path is None:
            raise ValueError("use_mask=True but mask_path is None")

        mask_path = Path(mask_path)
        mask_img = Image.open(mask_path).convert("L")

        # Resize to the latent's spatial resolution (note: this is the latent
        # H/W, not orig_height/orig_width).
        latent_h, latent_w = x_src.shape[2], x_src.shape[3]
        mask_img = mask_img.resize((latent_w, latent_h), resample=Image.NEAREST)

        mask = torch.from_numpy(np.array(mask_img)).to(device=device, dtype=torch.float32) / 255.0
        mask = (mask > 0.5).float()

        # Convention: mask==1 means "allow the update / keep delta_V", mask==0
        # means "block delta_V". To protect (block) the white region instead,
        # use: mask = 1.0 - mask
        mask = mask[None, None, :, :]  # [1,1,H,W]
        mask = mask.expand(x_src.shape[0], num_channels_latents, latent_h, latent_w)  # [B,C,H,W]

        # Pack to the same patch/token dimension as x_src_packed / V_delta_avg.
        mask_packed = pipe._pack_latents(
            mask, x_src.shape[0], num_channels_latents, latent_h, latent_w
        ).to(dtype=x_src_packed.dtype, device=x_src_packed.device)

    for i, t in tqdm(enumerate(timesteps)):

        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i = scheduler.sigmas[scheduler.step_index]
        if i < len(timesteps):
            t_im1 = scheduler.sigmas[scheduler.step_index + 1]
        else:
            t_im1 = t_i

        if T_steps - i > n_min:

            # Calculate the average of the V predictions
            V_delta_avg = torch.zeros_like(x_src_packed)

            for k in range(n_avg):

                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)

                zt_src = (1 - t_i) * x_src_packed + (t_i) * fwd_noise

                zt_tar = zt_edit + zt_src - x_src_packed

                # Merge in the future to avoid double computation
                Vt_src = calc_v_flux(
                    pipe,
                    latents=zt_src,
                    prompt_embeds=src_prompt_embeds,
                    pooled_prompt_embeds=src_pooled_prompt_embeds,
                    guidance=src_guidance,
                    text_ids=src_text_ids,
                    latent_image_ids=latent_src_image_ids,
                    t=t,
                )

                Vt_tar = calc_v_flux(
                    pipe,
                    latents=zt_tar,
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_tar_image_ids,
                    t=t,
                )

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            # ---- same-point semantic probe + wavelet low-frequency extraction + compensation ----
            t_val = float(t_i)
            semantic_weight = spectral_boost * (t_val ** 2)

            if semantic_weight > 1e-4:  # only compute when it actually matters, to save inference cost
                # Reuse zt_src from the last iteration of the loop above
                # (already (1-t_i)*x_src + t_i*eps).
                Vt_tar_at_src = calc_v_flux(
                    pipe,
                    latents=zt_src,  # key: evaluated at the same point as Vt_src
                    prompt_embeds=tar_prompt_embeds,
                    pooled_prompt_embeds=tar_pooled_prompt_embeds,
                    guidance=tar_guidance,
                    text_ids=tar_text_ids,
                    latent_image_ids=latent_src_image_ids,
                    t=t,
                )

                # Same-point semantic difference: removes the geometric confound,
                # yielding a cleaner semantic direction than V_delta.
                V_semantic = Vt_tar_at_src - Vt_src

                # Wavelet decomposition to extract the low-frequency component
                # (global semantics: color, pose).
                V_semantic_low = wavelet_extract_low_freq(
                    V_semantic,
                    pipe,
                    orig_height,
                    orig_width,
                    num_channels_latents,
                    n_levels=n_levels,
                )

                # Add into the ODE: original V_delta + semantic compensation
                # (active in the high-noise regime).
                V_delta_avg = V_delta_avg + semantic_weight * V_semantic_low

            if use_mask:
                mask_packed = expand_mask(mask_packed, kernel_size=3, iters=1)
                mask_packed = mask_packed.to(V_delta_avg.device)
                V_delta_avg = V_delta_avg * mask_packed  # block delta_V outside the mask

            # propagate direct ODE
            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

            
        else:  # i >= T_steps-n_min # regular sampling for last n_min steps

            if i == T_steps - n_min:
                # initialize SDEDIT-style generation phase
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                xt_src = scale_noise(scheduler, x_src_packed, t, noise=fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = calc_v_flux(
                pipe,
                latents=xt_tar,
                prompt_embeds=tar_prompt_embeds,
                pooled_prompt_embeds=tar_pooled_prompt_embeds,
                guidance=tar_guidance,
                text_ids=tar_text_ids,
                latent_image_ids=latent_tar_image_ids,
                t=t,
            )

            xt_tar = xt_tar.to(torch.float32)
            prev_sample = xt_tar + (t_im1 - t_i) * (Vt_tar)
            prev_sample = prev_sample.to(Vt_tar.dtype)
            xt_tar = prev_sample

    out = zt_edit if n_min == 0 else xt_tar
    unpacked_out = pipe._unpack_latents(
        out, orig_height, orig_width, pipe.vae_scale_factor
    )
    return unpacked_out

