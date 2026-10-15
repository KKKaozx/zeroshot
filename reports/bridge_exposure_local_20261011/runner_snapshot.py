"""Same eight training windows: Adapter updates or dropout controlled pair.

Local offline diagnosis, not simulation, gripper training, or generalization.
Frozen CLIP tokens are computed from actual TFRecord images once. Both arms use
Adapter-update mode uses eval in both arms. Dropout mode jointly trains both
arms, enabling the original p=0.1 Adapter dropout in one arm only.
"""
import argparse
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.update(USE_TF="0", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TF_CPP_MIN_LOG_LEVEL="2")
import numpy as np
import torch
from transformers import CLIPTokenizer
from dataset import UnifiedRobotDataset
from models import RobotAdapterModel
from diffusion_decoder import MultiscaleConditionalUnet1D
from replay_and_fit_bridge_positive import digest, evaluate, seed

ROOT = Path(__file__).resolve().parents[1]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--comparison", choices=["adapter_update", "dropout", "exposure"], default="adapter_update")
    ap.add_argument("--data", type=Path, default=Path("E:/dataset/bridge_v2_0.0.1/0.0.1"))
    ap.add_argument("--clip", type=Path, default=Path("D:/ntu_related/dissertation/hf_cache/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this bounded local control")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.monotonic()
    dropout_pair = args.comparison == "dropout"
    exposure = args.comparison == "exposure"
    arms = ["dropout_on"] if exposure else (["dropout_off", "dropout_on"] if dropout_pair else ["fixed_adapter", "joint_adapter"])
    report = dict(stage="local_current_adapter_"+args.comparison+"_pair", execution_completed=False,
                  model_parameters_updated=True, simulation_executed=False,
                  validation_evaluated=False, test_targets_used=False, gripper_trained=False,
                  downloads=False, gpu=torch.cuda.get_device_name(), torch=torch.__version__, groups={})
    source_names = ["models.py", "adapter.py", "diffusion_decoder.py", "dataset.py",
                    "fit_bridge_adapter_pair.py", "replay_and_fit_bridge_positive.py",
                    "evaluate_bridge_endpoint_metrics.py"]
    report["source_sha256"] = {name: file_sha(ROOT/"code"/name) for name in source_names}
    selection_report = ROOT/"reports/bridge_positive_local_20261011/report.json"
    previous = json.loads(selection_report.read_text())
    archive = ROOT.parent/"GPU cluster/bridge_diffusion_full_fit_v1.zip"
    if file_sha(archive) != previous["archive_sha256"]:
        raise ValueError("Previous positive control archive changed")
    with zipfile.ZipFile(archive) as z:
        bundle = torch.load(io.BytesIO(z.read("bridge_diffusion_full_fit_v1/full_context.pt")),
                            map_location="cpu", weights_only=False)
    train = bundle["data"]["train"]
    del bundle["data"]["validation"]
    selection = previous["selection"]
    indices = [row["cache_index"] for row in selection]
    targets = train["targets"][indices].clone()
    currents = train["current"][indices].clone()
    config = copy.deepcopy(bundle["config"])
    config["model"].update(name=str(args.clip), num_adapter_layers=8, attention_dim=512,
                           adapter_pooling="cls_patch_mean", dropout=0.1,
                           decoder_hidden_dim=256, diffusion_architecture="multiscale",
                           diffusion_down_dims=[256, 512, 1024])
    plan = dict(selection=selection, source_selection_sha256=file_sha(selection_report),
                config=config, updates_per_arm=500, batch_size=8, optimizer="Adam",
                learning_rate=3e-4, weight_decay=0, schedule="constant", gradient_clip=1.,
                checkpoints=[0, 20, 100, 500] if exposure else [0, 100, 500], training_noise_seed="420000+update, reset after context computation",
                comparison=args.comparison, arms=arms,
                adapter_mode=("Both jointly updated; off=eval, on=train with p=0.1; both evaluated in eval mode"
                              if dropout_pair else ("One jointly trained path; p=0.1 in training, eval mode for generation" if exposure else "eval in both arms, gradients enabled in joint arm")),
                dropout_seed="430000+update, independent of diffusion noise seed" if dropout_pair or exposure else None,
                sampling_steps=100, sampling_seeds=[1101, 1102, 1103],
                fit_thresholds=previous["pose_pass_criteria"], train_only=True,
                data_scope="Eight explicitly indexed records via current __getitem__; no full scanner/loader validation",
                changing_variable=("Number of optimizer updates along one training path; each window appears once per update" if exposure else "Whether Adapter dropout is enabled during joint optimization" if dropout_pair
                                   else "Whether current eight-layer Adapter parameters receive optimizer updates"))
    (args.output/"plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    report["plan"] = plan
    try:
        # Explicit indexing avoids a new scan of the entire dataset. The actual
        # current __getitem__ and TFRecord parser/prepare_image execute unchanged.
        data = UnifiedRobotDataset.__new__(UnifiedRobotDataset)
        data.chunk_size = 16
        data.bridge_gripper_policy = "reverse_scan_valid_steps_v2"
        data.bridge_current_gripper = "continuous"
        data.samples = []
        for row in selection:
            shard, record = row["episode"].split("::")
            data.samples.append(dict(source="tfrecord_bridge_state_action", file_path=str(args.data/shard),
                                     record_index=int(record), start_index=row["start"]))
            assert train["episode_keys"][row["cache_index"]] == row["episode"]
            assert int(train["starts"][row["cache_index"]]) == row["start"]
        examples = [data[i] for i in range(8)]
        images = torch.stack([x[1] for x in examples])
        raw_targets = torch.stack([x[3] for x in examples])
        raw_currents = torch.stack([x[2] for x in examples])
        masks = torch.stack([x[4] for x in examples])
        if not (torch.equal(targets, raw_targets) and torch.equal(currents, raw_currents)
                and bool(torch.all(masks[..., :7] == 1))):
            raise ValueError("Actual current raw parser does not match the same cached pose/current targets")
        instructions = [x[0] for x in examples]
        report["raw_data_check"] = dict(targets_exactly_match_previous_cache=True,
            currents_exactly_match_previous_cache=True, pose_masks_all_valid=True,
            image_sha256=tensor_sha(images), target_sha256=tensor_sha(targets), instructions=instructions)
        print("RAW_EIGHT_WINDOWS_MATCH_CACHE", flush=True)
        seed(42)
        model = RobotAdapterModel(config).cuda().eval()
        if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.adapter.modules()):
            raise ValueError("Train/eval dropout control assumes no Adapter BatchNorm")
        tokenizer = CLIPTokenizer.from_pretrained(str(args.clip), local_files_only=True)
        text = {k: v.cuda() for k, v in tokenizer(instructions, padding=True, truncation=True,
                                                return_tensors="pt").items()}
        with torch.no_grad():
            visual = model.vision_encoder(pixel_values=images.cuda()).last_hidden_state.detach()
            language = model.text_encoder(**text).last_hidden_state.detach()
            initial_context = model.adapter(visual, language, text["attention_mask"]).detach()
            actual_context = model.get_context_vector(images.cuda(), text["input_ids"], text["attention_mask"])
        error = float((actual_context-initial_context).abs().max())
        if error > 1e-5:
            raise ValueError("Cached token path differs from actual production get_context_vector")
        report["conditioning"] = dict(visual_shape=list(visual.shape), text_shape=list(language.shape),
            context_shape=list(initial_context.shape), production_context_max_abs_difference=error,
            visual_sha256=tensor_sha(visual), text_sha256=tensor_sha(language),
            initial_context_sha256=tensor_sha(initial_context))
        model.vision_encoder.cpu()
        model.text_encoder.cpu()
        frozen_clip_before = [digest(model.vision_encoder), digest(model.text_encoder)]
        seed(42)
        model.diffusion_decoder = MultiscaleConditionalUnet1D(action_dim=7, context_dim=1024,
            chunk_size=16, down_dims=(256, 512, 1024), diffusion_step_embed_dim=128,
            kernel_size=5, num_steps=100).cuda()
        decoder_initial = digest(model.diffusion_decoder)
        if decoder_initial != previous["decoder_initial_sha256"]:
            raise ValueError("Decoder initialization differs from the previous positive control")
        initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                   if not name.startswith(("vision_encoder.", "text_encoder."))}
        adapter_initial = digest(model.adapter)
        gripper_names = [name for name in initial if not name.startswith(("adapter.", "diffusion_decoder."))]
        report["initial_hashes"] = dict(adapter=adapter_initial, decoder=decoder_initial,
                                        clip=frozen_clip_before)
        targets_gpu, currents_gpu = targets.cuda(), currents.cuda()
        true = targets.numpy()
        sampler_seeds = plan["sampling_seeds"]
        for arm in arms:
            model.load_state_dict(initial, strict=False)
            if digest(model.adapter) != adapter_initial or digest(model.diffusion_decoder) != decoder_initial:
                raise ValueError("Pair does not share the same initial parameters")
            joint = dropout_pair or exposure or arm == "joint_adapter"
            dropout_enabled = arm == "dropout_on"
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith("diffusion_decoder.") or (joint and name.startswith("adapter.")))
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=3e-4, weight_decay=0)
            history, losses, durations = [], [], []
            first_grad = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            def context():
                return model.adapter(visual, language, text["attention_mask"]) if joint else initial_context
            def check(update):
                model.eval()
                with torch.no_grad():
                    ctx = context().detach()
                    pred, metrics = evaluate(model, ctx, currents_gpu, true, sampler_seeds, batch=8)
                history.append(dict(update=update, elapsed_seconds=time.monotonic()-started,
                    context_change_rms=float((ctx-initial_context).square().mean().sqrt()), **metrics))
                (args.output/f"{arm}_history.json").write_text(json.dumps(history, indent=2)+"\n")
                if exposure:
                    np.savez_compressed(args.output/f"update_{update}_predictions.npz", predictions=pred,
                                        targets=true, sampling_seeds=sampler_seeds)
                print("ADAPTER_PAIR", arm, update, json.dumps(metrics), flush=True)
                return pred, metrics, ctx
            finalpred, final, finalctx = check(0)
            for update in range(1, 501):
                model.train()
                model.adapter.train(dropout_enabled)
                model.diffusion_decoder.train()
                torch.cuda.synchronize()
                tick = time.monotonic()
                if dropout_pair or exposure:
                    seed(430000+update)
                ctx = context()
                seed(420000+update)  # Identical timesteps/noise, independent of Adapter forward.
                output, truth, _ = model.diffusion_loss(targets_gpu, ctx, currents_gpu)
                loss = torch.nn.functional.mse_loss(output, truth)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if first_grad is None:
                    first_grad = {name: any(p.grad is not None and bool(torch.any(p.grad != 0))
                        for p in module.parameters()) for name, module in
                        [("adapter", model.adapter), ("decoder", model.diffusion_decoder),
                         ("clip_vision", model.vision_encoder), ("clip_text", model.text_encoder)]}
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                durations.append(time.monotonic()-tick)
                losses.append(float(loss.detach()))
                if update == 1 or update % 100 == 0:
                    print("PAIR_UPDATE", arm, update, "loss", losses[-1], flush=True)
                if update in plan["checkpoints"]:
                    finalpred, final, finalctx = check(update)
            non_pose_unchanged = all(torch.equal(model.state_dict()[n].detach().cpu(), initial[n]) for n in gripper_names)
            adapter_changed = digest(model.adapter) != adapter_initial
            if not non_pose_unchanged or adapter_changed != joint or digest(model.diffusion_decoder) == decoder_initial:
                raise ValueError("Parameter mutation scope failed")
            if first_grad["adapter"] != joint or not first_grad["decoder"] or first_grad["clip_vision"] or first_grad["clip_text"]:
                raise ValueError("Gradient scope failed")
            np.savez_compressed(args.output/f"{arm}_predictions.npz", predictions=finalpred, targets=true,
                                sampling_seeds=sampler_seeds, context=finalctx.cpu().numpy())
            report["groups"][arm] = dict(final_metrics=final, history=history, updates=500,
                first_gradients=first_grad, adapter_changed=adapter_changed, decoder_changed=True,
                adapter_dropout_enabled_during_training=dropout_enabled,
                frozen_gripper_unchanged=non_pose_unchanged, mean_warm_update_seconds=float(np.mean(durations[1:])),
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                last_100_mean_loss=float(np.mean(losses[-100:])), adapter_final_sha256=digest(model.adapter),
                decoder_final_sha256=digest(model.diffusion_decoder))
            (args.output/f"{arm}_losses.json").write_text(json.dumps(losses)+"\n")
            del optimizer, params, ctx, output, truth, loss
            for p in model.parameters():
                p.grad = None
            gc.collect()
            torch.cuda.empty_cache()
        if [digest(model.vision_encoder), digest(model.text_encoder)] != frozen_clip_before:
            raise ValueError("Frozen CLIP parameters changed")
        if any(file_sha(ROOT/"code"/name) != value for name, value in report["source_sha256"].items()):
            raise ValueError("Source changed during execution")
        report.update(execution_completed=True, clip_unchanged=True, elapsed_seconds=time.monotonic()-started,
                      limits=["Eight training windows, no validation or robot task-success claim.",
                             ("Both arms jointly trained; only dropout mode changed; original full-data optimizer not reproduced."
                              if dropout_pair else ("One small-set training curve; update count is not an isolated explanation of earlier full-data failure." if exposure else "Adapter dropout disabled in both arms; this does not reproduce joint training with dropout.")),
                             "Only pose loss optimized; frozen gripper predictions do not establish gripper skill.",
                             "One shared training initialization, three sampling seeds, not three training seeds.",
                             "No large checkpoint retained; predictions/settings/hashes retained for bounded reproduction."])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        (args.output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
        print("CURRENT ADAPTER PAIR REPORT", args.output/"report.json", flush=True)


if __name__ == "__main__":
    main()
