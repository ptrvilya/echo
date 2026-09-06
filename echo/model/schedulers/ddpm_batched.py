import torch
from diffusers.schedulers import DDPMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.utils.torch_utils import randn_tensor


class DDPMSchedulerBatched(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _get_variance(
        self, t: int | torch.IntTensor, predicted_variance=None, variance_type=None
    ):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = torch.where(
            prev_t >= 0, self.alphas_cumprod[prev_t], self.one
        )
        # alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        guidance: torch.FloatTensor | None = None,
        return_dict: bool = True,
    ) -> DDPMSchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Add guidance signal on step 5 if provided.

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            guidance (`torch.FloatTensor`, *optional*):
                A tensor with per-element gradients from the classifier-based guidance.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        n_examples = model_output.size(0)
        if isinstance(timestep, int):
            timestep = torch.tensor([timestep]).expand(n_examples)
        elif isinstance(timestep, torch.Tensor) and timestep.numel() == 1:
            timestep = timestep.expand(n_examples)
        t = timestep

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in [
            "learned",
            "learned_range",
        ]:
            model_output, predicted_variance = torch.split(
                model_output, sample.shape[1], dim=1
            )
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        # alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        alpha_prod_t_prev = torch.where(
            prev_t >= 0, self.alphas_cumprod[prev_t], self.one
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        _ndim = model_output.ndim - 1
        _target_shape = [-1] + _ndim * [1]
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t.reshape(_target_shape) ** (0.5) * model_output
            ) / alpha_prod_t.reshape(_target_shape) ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v_prediction":
            _ndim = model_output.ndim - 1
            pred_original_sample = (
                alpha_prod_t.reshape(_target_shape) ** 0.5
            ) * sample - (beta_prod_t.reshape(_target_shape) ** 0.5) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )

        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        device = model_output.device
        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        pred_original_sample_coeff = pred_original_sample_coeff.to(device)
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t
        current_sample_coeff = current_sample_coeff.to(device)

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = (
            pred_original_sample_coeff.reshape(_target_shape) * pred_original_sample
            + current_sample_coeff.reshape(_target_shape) * sample
        )

        # 6. Add noise
        device = model_output.device
        variance_zero = torch.zeros(
            model_output.shape, device=device, dtype=model_output.dtype
        )
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=device,
            dtype=model_output.dtype,
        )
        variance = (
            self._get_variance(t, predicted_variance=predicted_variance)
            .to(device)
            .reshape(_target_shape)
        )
        if self.variance_type == "fixed_small_log":
            variance = variance * variance_noise
        elif self.variance_type == "learned_range":
            variance = torch.exp(0.5 * variance) * variance_noise
        else:
            variance = (variance**0.5) * variance_noise
        variance[t == 0] = variance_zero[t == 0]

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample
        )

    def previous_timestep(self, timestep: torch.IntTensor):
        if self.custom_timesteps or self.num_inference_steps:
            # index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            # if isinstance(timestep, int):
            #     timestep = torch.tensor([timestep])
            _, index = (self.timesteps.unsqueeze(0) == timestep.unsqueeze(1)).nonzero(
                as_tuple=True
            )
            prev_t = torch.where(
                index == self.timesteps.shape[0] - 1,
                torch.tensor(-1),
                self.timesteps[torch.clamp(index + 1, 0, self.timesteps.shape[0] - 1)],
            )

            # prev_t = torch.where(index == self.timesteps.shape[0] - 1, torch.tensor(-1), self.timesteps[index + 1])
            # if index == self.timesteps.shape[0] - 1:
            #     prev_t = torch.tensor(-1)
            # else:
            #     prev_t = self.timesteps[index + 1]
        else:
            prev_t = timestep - 1
        return prev_t
