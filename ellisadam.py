# Apache 2.0 licenced from https://github.com/seal-rg/recurrent-pretraining/blob/main/LICENSE
"""
                                    Apache License
                        Version 2.0, January 2004
                        http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

    "License" shall mean the terms and conditions for use, reproduction,
    and distribution as defined by Sections 1 through 9 of this document.

    "Licensor" shall mean the copyright owner or entity authorized by
    the copyright owner that is granting the License.

    "Legal Entity" shall mean the union of the acting entity and all
    other entities that control, are controlled by, or are under common
    control with that entity. For the purposes of this definition,
    "control" means (i) the power, direct or indirect, to cause the
    direction or management of such entity, whether by contract or
    otherwise, or (ii) ownership of fifty percent (50%) or more of the
    outstanding shares, or (iii) beneficial ownership of such entity.

    "You" (or "Your") shall mean an individual or Legal Entity
    exercising permissions granted by this License.

    "Source" form shall mean the preferred form for making modifications,
    including but not limited to software source code, documentation
    source, and configuration files.

    "Object" form shall mean any form resulting from mechanical
    transformation or translation of a Source form, including but
    not limited to compiled object code, generated documentation,
    and conversions to other media types.

    "Work" shall mean the work of authorship, whether in Source or
    Object form, made available under the License, as indicated by a
    copyright notice that is included in or attached to the work
    (an example is provided in the Appendix below).

    "Derivative Works" shall mean any work, whether in Source or Object
    form, that is based on (or derived from) the Work and for which the
    editorial revisions, annotations, elaborations, or other modifications
    represent, as a whole, an original work of authorship. For the purposes
    of this License, Derivative Works shall not include works that remain
    separable from, or merely link (or bind by name) to the interfaces of,
    the Work and Derivative Works thereof.

    "Contribution" shall mean any work of authorship, including
    the original version of the Work and any modifications or additions
    to that Work or Derivative Works thereof, that is intentionally
    submitted to Licensor for inclusion in the Work by the copyright owner
    or by an individual or Legal Entity authorized to submit on behalf of
    the copyright owner. For the purposes of this definition, "submitted"
    means any form of electronic, verbal, or written communication sent
    to the Licensor or its representatives, including but not limited to
    communication on electronic mailing lists, source code control systems,
    and issue tracking systems that are managed by, or on behalf of, the
    Licensor for the purpose of discussing and improving the Work, but
    excluding communication that is conspicuously marked or otherwise
    designated in writing by the copyright owner as "Not a Contribution."

    "Contributor" shall mean Licensor and any individual or Legal Entity
    on behalf of whom a Contribution has been received by Licensor and
    subsequently incorporated within the Work.

2. Grant of Copyright License. Subject to the terms and conditions of
    this License, each Contributor hereby grants to You a perpetual,
    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    copyright license to reproduce, prepare Derivative Works of,
    publicly display, publicly perform, sublicense, and distribute the
    Work and such Derivative Works in Source or Object form.

3. Grant of Patent License. Subject to the terms and conditions of
    this License, each Contributor hereby grants to You a perpetual,
    worldwide, non-exclusive, no-charge, royalty-free, irrevocable
    (except as stated in this section) patent license to make, have made,
    use, offer to sell, sell, import, and otherwise transfer the Work,
    where such license applies only to those patent claims licensable
    by such Contributor that are necessarily infringed by their
    Contribution(s) alone or by combination of their Contribution(s)
    with the Work to which such Contribution(s) was submitted. If You
    institute patent litigation against any entity (including a
    cross-claim or counterclaim in a lawsuit) alleging that the Work
    or a Contribution incorporated within the Work constitutes direct
    or contributory patent infringement, then any patent licenses
    granted to You under this License for that Work shall terminate
    as of the date such litigation is filed.

4. Redistribution. You may reproduce and distribute copies of the
    Work or Derivative Works thereof in any medium, with or without
    modifications, and in Source or Object form, provided that You
    meet the following conditions:

    (a) You must give any other recipients of the Work or
        Derivative Works a copy of this License; and

    (b) You must cause any modified files to carry prominent notices
        stating that You changed the files; and

    (c) You must retain, in the Source form of any Derivative Works
        that You distribute, all copyright, patent, trademark, and
        attribution notices from the Source form of the Work,
        excluding those notices that do not pertain to any part of
        the Derivative Works; and

    (d) If the Work includes a "NOTICE" text file as part of its
        distribution, then any Derivative Works that You distribute must
        include a readable copy of the attribution notices contained
        within such NOTICE file, excluding those notices that do not
        pertain to any part of the Derivative Works, in at least one
        of the following places: within a NOTICE text file distributed
        as part of the Derivative Works; within the Source form or
        documentation, if provided along with the Derivative Works; or,
        within a display generated by the Derivative Works, if and
        wherever such third-party notices normally appear. The contents
        of the NOTICE file are for informational purposes only and
        do not modify the License. You may add Your own attribution
        notices within Derivative Works that You distribute, alongside
        or as an addendum to the NOTICE text from the Work, provided
        that such additional attribution notices cannot be construed
        as modifying the License.

    You may add Your own copyright statement to Your modifications and
    may provide additional or different license terms and conditions
    for use, reproduction, or distribution of Your modifications, or
    for any such Derivative Works as a whole, provided Your use,
    reproduction, and distribution of the Work otherwise complies with
    the conditions stated in this License.

5. Submission of Contributions. Unless You explicitly state otherwise,
    any Contribution intentionally submitted for inclusion in the Work
    by You to the Licensor shall be under the terms and conditions of
    this License, without any additional terms or conditions.
    Notwithstanding the above, nothing herein shall supersede or modify
    the terms of any separate license agreement you may have executed
    with Licensor regarding such Contributions.

6. Trademarks. This License does not grant permission to use the trade
    names, trademarks, service marks, or product names of the Licensor,
    except as required for reasonable and customary use in describing the
    origin of the Work and reproducing the content of the NOTICE file.

7. Disclaimer of Warranty. Unless required by applicable law or
    agreed to in writing, Licensor provides the Work (and each
    Contributor provides its Contributions) on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
    implied, including, without limitation, any warranties or conditions
    of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
    PARTICULAR PURPOSE. You are solely responsible for determining the
    appropriateness of using or redistributing the Work and assume any
    risks associated with Your exercise of permissions under this License.

8. Limitation of Liability. In no event and under no legal theory,
    whether in tort (including negligence), contract, or otherwise,
    unless required by applicable law (such as deliberate and grossly
    negligent acts) or agreed to in writing, shall any Contributor be
    liable to You for damages, including any direct, indirect, special,
    incidental, or consequential damages of any character arising as a
    result of this License or out of the use or inability to use the
    Work (including but not limited to damages for loss of goodwill,
    work stoppage, computer failure or malfunction, or any and all
    other commercial damages or losses), even if such Contributor
    has been advised of the possibility of such damages.

9. Accepting Warranty or Additional Liability. While redistributing
    the Work or Derivative Works thereof, You may choose to offer,
    and charge a fee for, acceptance of support, warranty, indemnity,
    or other liability obligations and/or rights consistent with this
    License. However, in accepting such obligations, You may act only
    on Your own behalf and on Your sole responsibility, not on behalf
    of any other Contributor, and only if You agree to indemnify,
    defend, and hold each Contributor harmless for any liability
    incurred by, or claims asserted against, such Contributor by reason
    of your accepting any such warranty or additional liability.

END OF TERMS AND CONDITIONS

APPENDIX: How to apply the Apache License to your work.

    To apply the Apache License to your work, attach the following
    boilerplate notice, with the fields enclosed by brackets "[]"
    replaced with your own identifying information. (Don't include
    the brackets!)  The text should be enclosed in the appropriate
    comment syntax for the file format. We also recommend that a
    file or class name and description of purpose be included on the
    same "printed page" as the copyright notice for easier
    identification within third-party archives.

Copyright [2023] Lightning AI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Copyright [2025] Jonas Geiping, John Kirchenbauer, Sean McLeish, Khalid Saifullah, Manli Shu, Neel Jain, Siddarth Singh, Abhimanyu Hans, Monte Hoover and Prajwal Singhanaia.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
from torch.optim import Optimizer
from torch import Tensor
from typing import List, Optional, Tuple, Union
import copy
from math import sqrt

def _parse_str_to_dtype(string_rep: str):
    if "bf16" in string_rep:
        return torch.bfloat16
    elif "f16" in string_rep or "fp16" in string_rep:
        return torch.float16
    else:
        return torch.float32

    
class ELLISAdam(Optimizer):
    # https://github.com/seal-rg/recurrent-pretraining/blob/main/recpre/optim.py#L728
    def __init__(
        self,
        params,
        lr: Union[float, torch.Tensor] = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        *,
        foreach: Optional[bool] = None,
        nesterov: bool = False,
        eps_adjustment: bool = False,
        update_clipping: bool = False,
        kahan_sum_compensation: bool = False,
        buffer_dtype: Optional[Union[torch.dtype, str]] = None,  # can be torch.float16 / torch.bfloat16
        running_init: bool = False,
        tensor_wise_finite_check: bool = False,
        tensor_wise_gradient_normalization: bool = False,
        adafactor_like_beta_corrections: bool = False,
        atan_adam: bool = False,
        decouple_wd: bool = True,
        brute_force_clip: Optional[float] = None,
        poly_ema_p: Optional[float] = None,
    ):
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32),
            init_lr=copy.deepcopy(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            nesterov=nesterov,
            eps_adjustment=eps_adjustment,
            update_clipping=update_clipping,
            kahan_sum_compensation=kahan_sum_compensation,
            buffer_dtype=_parse_str_to_dtype(buffer_dtype) if isinstance(buffer_dtype, str) else buffer_dtype,
            running_init=running_init,
            tensor_wise_finite_check=tensor_wise_finite_check,
            tensor_wise_gradient_normalization=tensor_wise_gradient_normalization,
            adafactor_like_beta_corrections=adafactor_like_beta_corrections,
            atan_adam=atan_adam,
            decouple_wd=decouple_wd,
            brute_force_clip=brute_force_clip,
            poly_ema_p=poly_ema_p,
        )
        self.arg_lr = lr
        if foreach:
            raise ValueError("Todo: reinstate a foreach version, minimizing additional mem alloc")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=torch.float32)

    @torch.no_grad()
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        kahan_comps,
        running_init: bool = False,
        buffer_dtype=None,
        kahan_sum_compensation: bool = False,
        tensor_wise_gradient_normalization: bool = False,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                _tensor_constructors = dict(memory_format=torch.preserve_format)
                if buffer_dtype is not None:
                    _tensor_constructors["dtype"] = buffer_dtype

                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0, dtype=torch.long)

                if kahan_sum_compensation:
                    state["kahan_comps"] = torch.zeros_like(p, **_tensor_constructors)
                else:
                    state["kahan_comps"] = None
                if running_init:
                    grad = p.grad if not tensor_wise_gradient_normalization else p.grad / p.grad.norm()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.clone().to(**_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.pow(2).clone().to(**_tensor_constructors)
                else:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, **_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, **_tensor_constructors)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
            kahan_comps.append(state["kahan_comps"])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            kahan_comps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                running_init=group["running_init"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
            )
            _single_tensor_modded_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                init_lr=group["init_lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                nesterov=group["nesterov"],
                eps_adjustment=group["eps_adjustment"],
                update_clipping=group["update_clipping"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_finite_check=group["tensor_wise_finite_check"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
                adafactor_like_beta_corrections=group["adafactor_like_beta_corrections"],
                atan_adam=group["atan_adam"],
                decouple_wd=group["decouple_wd"],
                brute_force_clip=group["brute_force_clip"],
                poly_ema_p=group["poly_ema_p"],
            )

        return loss


def _single_tensor_modded_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    kahan_comps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    init_lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    nesterov: bool = False,
    eps_adjustment: bool = False,
    update_clipping: bool = False,
    kahan_sum_compensation: bool = False,
    buffer_dtype=Optional[torch.dtype],
    tensor_wise_finite_check: bool = False,
    tensor_wise_gradient_normalization: bool = False,
    adafactor_like_beta_corrections: bool = False,
    atan_adam: bool = False,
    decouple_wd: bool = False,
    brute_force_clip: Optional[float] = None,
    poly_ema_p: Optional[float] = None,
):
    if adafactor_like_beta_corrections:
        # update group step
        step_t = state_steps[0]  # crime
        step_t += 1
        beta1 = (beta1**step_t - beta1) / (beta1**step_t - 1)
        beta2 = (beta2**step_t - beta2) / (beta2**step_t - 1)

    if poly_ema_p is not None:
        step_t = state_steps[0]  # crime
        # beta1 = step_t / (step_t + poly_ema_p)
        beta2 = step_t / (step_t + poly_ema_p)  # palm: 1 - step_t ** -0.8

    if nesterov:
        alpha = 2 * (1 - beta1) - (1 - beta1) ** 2  # only for nesterov to fuse the two lerps

    for i, param in enumerate(params):
        grad = grads[i].to(buffer_dtype)
        if tensor_wise_finite_check and (~torch.isfinite(grad)).sum() > 0:
            continue

        if tensor_wise_gradient_normalization:
            grad = grad / grad.norm()
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        kahan_comp = kahan_comps[i]

        # Decay the first and second moment running average coefficient
        if nesterov:
            # Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            # We undo these ops later on, which could cause numerical issues but saves
            # us from having to make an extra copy of the gradients.
            exp_avg.lerp_(grad, alpha)
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr.clone() if isinstance(lr, torch.Tensor) else lr

        if update_clipping:
            rms = grad.pow(2).div_(exp_avg_sq.clamp_(min=eps**2)).mean().sqrt()  # impl like optimi
            step_size = step_size / rms.clamp(min=1.0)

        if not adafactor_like_beta_corrections:
            step_t += 1
            bias_correction1 = 1 - beta1**step_t
            bias_correction2 = 1 - beta2**step_t
            bias_correction2_sqrt = sqrt(bias_correction2)

            step_size = step_size / bias_correction1
        else:
            bias_correction2 = 1.0
            bias_correction2_sqrt = 1.0

        # Actual adam step
        if kahan_sum_compensation:
            # Perform stepweight decay
            if decouple_wd:
                kahan_comp.mul_(1 - lr / init_lr * weight_decay)
            else:
                kahan_comp.mul_(1 - lr * weight_decay)
            if atan_adam:
                # a = b = 1
                kahan_comp.add_(torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt)), alpha=-step_size)
            elif eps_adjustment:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)
            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)
            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # Perform stepweight decay
            if decouple_wd:
                param.mul_(1 - lr / init_lr * weight_decay)
            else:
                param.mul_(1 - lr * weight_decay)
            if atan_adam:
                update = torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt))
                if brute_force_clip is not None:
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.add_(update, alpha=-step_size)
            elif eps_adjustment:
                if brute_force_clip is not None:
                    update = exp_avg / exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt()
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                if brute_force_clip is not None:
                    update = exp_avg / exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)

        # undo nadam
        if nesterov:
            exp_avg.lerp_(grad, 1 - 1 / beta1)