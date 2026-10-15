"""Isolated xyz normalization: core model source remains frozen."""
import torch
from models import RobotAdapterModel

class NormalizedRobotAdapterModel(RobotAdapterModel):
    def __init__(self, config):
        super().__init__(config)
        self.xyz_normalizer = config.get("xyz_normalizer", {"scale": [1.,1.,1.], "offset": [0.,0.,0.]})
        if len(self.xyz_normalizer["scale"]) != 3 or len(self.xyz_normalizer["offset"]) != 3 or not all(float(v)>0 for v in self.xyz_normalizer["scale"]):
            raise ValueError("Three positive axis scales required")
        self.identity_xyz = self.xyz_normalizer["scale"] == [1.,1.,1.] and self.xyz_normalizer["offset"] == [0.,0.,0.]
        if self.decoder_type == "diffusion" and self.diffusion_prediction_type != "sample":
            raise ValueError("This control is x0-only")

    def convert(self, actions, inverse=False):
        scale = actions.new_tensor(self.xyz_normalizer["scale"])
        offset = actions.new_tensor(self.xyz_normalizer["offset"])
        xyz = (actions[..., :3]-offset)/scale if inverse else actions[..., :3]*scale+offset
        return torch.cat([xyz, actions[..., 3:]], dim=-1)

    def predict_gripper_logits(self, context, pose_actions, current_gripper=None):
        physical = pose_actions if self.identity_xyz else self.convert(pose_actions, inverse=True)
        return super().predict_gripper_logits(context, physical, current_gripper)

    def diffusion_loss(self, actions, context, current_gripper=None):
        if self.identity_xyz:
            return super().diffusion_loss(actions, context, current_gripper)
        if actions.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError("Unexpected action shape")
        target = self.convert(actions)[..., :7]
        ts = torch.randint(0,self.num_diffusion_steps,(actions.shape[0],),device=actions.device)
        noise = torch.randn_like(target)
        abar = self.alpha_bars.gather(0,ts).reshape(-1,1,1)
        noisy = abar.sqrt()*target+(1-abar).sqrt()*noise
        predicted = self.diffusion_decoder(noisy,ts,context)
        # Physical-unit MSE implements normalized xyz error weighted by 1/scale**2.
        # Gripper and loss targets use the original tensors, without roundtrip drift.
        logits = RobotAdapterModel.predict_gripper_logits(self,context,actions[..., :7],current_gripper)
        return self.convert(predicted,inverse=True), actions[..., :7], logits

    def clip_encoded_clean(self, clean):
        physical = self.convert(clean, inverse=True)
        limit = float(self.max_normalized_position or 3.)
        physical = torch.cat([physical[..., :3].clamp(-limit,limit), physical[..., 3:].clamp(-1,1)], dim=-1)
        return self.convert(physical)

    @torch.no_grad()
    def sample(self, context, current_gripper=None):
        if self.identity_xyz:
            return super().sample(context, current_gripper)
        if self.decoder_type != "diffusion" or not self.separate_gripper_head or self.gripper_target_mode != "state":
            raise ValueError("Frozen diffusion/state gripper control required")
        actions = torch.randn(context.shape[0], self.chunk_size, 7, device=context.device)
        for step in reversed(range(self.num_diffusion_steps)):
            ts = torch.full((context.shape[0],), step, device=context.device, dtype=torch.long)
            x0 = self.diffusion_decoder(actions, ts, context)
            alpha, abar = self.alphas[step], self.alpha_bars[step]
            epsilon = (actions-abar.sqrt()*x0)/(1-abar).sqrt()
            mean = (actions-(1-alpha)*epsilon/(1-abar).sqrt())/alpha.sqrt()
            if self.clip_denoised:
                clean = (actions-(1-abar).sqrt()*epsilon)/abar.sqrt()
                clean = self.clip_encoded_clean(clean)
                mean = self.posterior_mean_x0[step]*clean+self.posterior_mean_xt[step]*actions
            actions = mean+self.posterior_variance[step].sqrt()*torch.randn_like(actions) if step>0 else mean
        actions = self.convert(actions, inverse=True)
        if self.max_normalized_position is not None:
            actions[..., :3] = actions[..., :3].clamp(-float(self.max_normalized_position),float(self.max_normalized_position))
        q = actions[..., 3:7]
        identity = torch.zeros_like(q); identity[..., 3] = 1.
        actions[..., 3:7] = torch.where(q.norm(dim=-1,keepdim=True)>1e-6,torch.nn.functional.normalize(q,dim=-1),identity)
        logits = RobotAdapterModel.predict_gripper_logits(self,context,actions,current_gripper)
        gripper = torch.where(logits>=0,torch.ones_like(logits),-torch.ones_like(logits))
        return torch.cat([actions,gripper.unsqueeze(-1)],dim=-1)
