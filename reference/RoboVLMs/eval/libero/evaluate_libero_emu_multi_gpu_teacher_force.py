"""
Evaluate UniVLA / RoboVLMs-CoT via LIBERO with multi-GPU episode-level sharding,
using teacher-forced subgoal images from the training set.
"""
import os
import faulthandler
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"
if os.environ.get("MUJOCO_GL", "").lower() == "egl":
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
faulthandler.enable()
import argparse
import json
import logging
import pickle
import random
import sys
import time
import traceback
from pathlib import Path
import tqdm
import cv2
import numpy as np
import torch
import torch.distributed as dist
from pytorch_lightning import seed_everything
from datetime import timedelta
sys.path.insert(0, Path(__file__).absolute().parents[2].as_posix())

from model_wrapper_emu import EmuVLAModel
from libero_utils import (
    get_libero_camera_image,
    get_episode_length,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_gif,
)

sys.path.append("/inspire/hdd/project/socialsimulation/chenfangke-253108540237/tsli/LIBERO")
from libero.libero import benchmark

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s - %(name)s - %(levelname)s - %(message)s]"
)
logger = logging.getLogger(__name__)


def normalize_task_text(text: str) -> str:
    text = " ".join(text.strip().split()).lower()
    return text.rstrip(".")


def resolve_relative_path(path: str, roots: list[str]) -> str:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    for root in roots:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return candidate
    return path


class TeacherForcingSampler:
    def __init__(
        self,
        data_path: str,
        min_h: int,
        max_h: int,
        seed: int,
        search_roots: list[str],
        image_key: str = "image",
    ):
        self.min_h = min_h
        self.max_h = max_h
        self.rng = random.Random(seed)
        self.search_roots = search_roots
        self.image_key = image_key
        self.scene_index = {}
        self.current_scene = None

        with open(data_path, "rb") as f:
            data = pickle.load(f)
        for scene in data:
            key = normalize_task_text(scene["text"])
            self.scene_index.setdefault(key, []).append(scene)

        if not self.scene_index:
            raise ValueError(f"No scenes loaded from {data_path}")

    def reset_episode(self, task_description: str):
        key = normalize_task_text(task_description)
        scenes = self.scene_index.get(key)
        if not scenes:
            raise KeyError(f"Task description not found in training set: {task_description}")
        self.current_scene = self.rng.choice(scenes)

    def get_goal_tokens(self, step_idx: int):
        if self.current_scene is None:
            raise RuntimeError("TeacherForcingSampler.reset_episode must be called before get_goal_tokens.")
        u = self.rng.randint(self.min_h, self.max_h)
        image_paths = self.current_scene[self.image_key]
        if not image_paths:
            raise ValueError(f"Current scene has no image paths for key '{self.image_key}'.")
        goal_idx = min(max(step_idx + u, 0), len(image_paths) - 1)
        path = resolve_relative_path(image_paths[goal_idx], self.search_roots)
        goal_tokens = np.load(path)
        if goal_tokens.ndim == 2:
            goal_tokens = goal_tokens[None, ...]
        return goal_tokens, goal_idx, u


def world_info_from_env():
    local_rank = 0
    for v in (
        "LOCAL_RANK",
        "MPI_LOCALRANKID",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
    ):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ("RANK", "PMI_RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ("WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS", "OMPI_COMM_WORLD_SIZE"):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def resolve_physical_cuda_device(visible_index: int) -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cuda_visible:
        return visible_index
    if cuda_visible.isdigit():
        if visible_index != 0:
            raise RuntimeError(
                f"Requested visible device index {visible_index} but "
                f"CUDA_VISIBLE_DEVICES={cuda_visible} exposes only one device."
            )
        return int(cuda_visible)
    device_ids = [int(x) for x in cuda_visible.split(",") if x.strip() != ""]
    if visible_index < 0 or visible_index >= len(device_ids):
        raise RuntimeError(
            f"Requested visible device index {visible_index} but "
            f"CUDA_VISIBLE_DEVICES={cuda_visible} provides {len(device_ids)} device(s)."
        )
    return device_ids[visible_index]


def setup_distributed():
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=240) # 4 hours
        )
    local_rank, rank, world_size = world_info_from_env()
    # Bind EGL to the physical device that matches this process's visible index.
    render_gpu_device_id = resolve_physical_cuda_device(local_rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)
    
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size, render_gpu_device_id


def prepare_observation(obs):
    img = get_libero_image(obs)
    observation = {
        "full_image": img,
        "state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        ),
    }
    if "robot0_eye_in_hand_image" in obs:
        observation["wrist_image"] = get_libero_wrist_image(obs)
    return observation, img


def obs_key_to_camera_name(obs_key):
    if not obs_key.endswith("_image"):
        raise ValueError(f"Unexpected observation key format: {obs_key}")
    return obs_key[: -len("_image")]


def episode_is_assigned(task_id, episode_idx, num_trials_per_task, rank, world_size):
    global_episode_idx = task_id * num_trials_per_task + episode_idx
    return global_episode_idx % world_size == rank, global_episode_idx


def evaluate(
    model,
    task_suite_name,
    local_log_dir,
    rank,
    world_size,
    render_gpu_device_id,
    num_trials_per_task,
    num_steps_wait,
    teacher_sampler=None,
    debug=False,
    teacher_force_mode="cot",
    perspective_obs_key="robot0_eye_in_hand_image",
    camera_resolution=256,
):
    os.makedirs(local_log_dir, exist_ok=True)
    log_path = os.path.join(local_log_dir, f"eval_rank{rank}.txt")
    log_file = open(log_path, "w")
    logger.info("Logging to %s", log_path)

    results_path = os.path.join(local_log_dir, f"episodes_rank{rank}.jsonl")
    results_file = open(results_path, "w")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    EP_LEN = get_episode_length(task_suite_name)

    logger.info("Task suite: %s", task_suite_name)
    log_file.write(f"Task suite: {task_suite_name}\n")
    log_file.write(f"World size: {world_size}, rank: {rank}\n")
    log_file.write(f"Num trials per task: {num_trials_per_task}\n")
    log_file.flush()

    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc=f"Task Pregress", total=num_tasks_in_suite, disable=(rank != 0 )):
        assigned_episode_indices = []
        for episode_idx in range(num_trials_per_task):
            assigned, _ = episode_is_assigned(
                task_id, episode_idx, num_trials_per_task, rank, world_size
            )
            if assigned:
                assigned_episode_indices.append(episode_idx)

        if not assigned_episode_indices:
            continue

        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        if len(initial_states) < num_trials_per_task:
            raise ValueError(
                f"Task {task_id} has only {len(initial_states)} initial states, "
                f"but num_trials_per_task={num_trials_per_task}"
            )

        camera_names = ["agentview"]
        if getattr(model, "use_gripper", False):
            camera_names.append("robot0_eye_in_hand")
        if teacher_force_mode == "perspective":
            target_camera = obs_key_to_camera_name(perspective_obs_key)
            if target_camera not in camera_names:
                camera_names.append(target_camera)

        env, task_description = get_libero_env(
            task,
            resolution=camera_resolution,
            render_gpu_device_id=render_gpu_device_id,
            camera_names=camera_names,
        )
        task_episodes = 0
        task_successes = 0

        logger.info("Task %s: %s", task_id, task_description)
        log_file.write(f"\nTask {task_id}: {task_description}\n")
        log_file.write(
            f"Assigned episodes: {assigned_episode_indices}\n"
        )
        log_file.flush()

        for episode_idx in tqdm.tqdm(assigned_episode_indices, desc=f"Episode Pregress", total=len(assigned_episode_indices), disable=(rank != 0 )):
            env.reset()
            model.reset()

            obs = env.set_init_state(initial_states[episode_idx])
            t = 0
            replay_images = []
            done = False
            error = None

            if model.use_cot:
                thought = [""]
            if teacher_sampler is not None:
                teacher_sampler.reset_episode(task_description)

            logger.info("Starting episode %s (task %s)", episode_idx + 1, task_id)
            log_file.write(
                f"Starting episode {episode_idx + 1} (task {task_id})\n"
            )
            log_file.flush()

            action_counter = 0
            while t < EP_LEN + num_steps_wait:
                try:
                    if t < num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue
                    # if rank == 0:
                    #     logger.info("Episode %s (Task %s): Step %s", episode_idx + 1, task_id, t)
                    
                    observation, img = prepare_observation(obs)
                    if debug:
                        if model.use_cot:
                            text_img = np.ones((img.shape[0], 1000, 3), dtype=np.uint8) * 255
                            lines = thought[0].replace("@", "\n").split("\n")
                            for i, line in enumerate(lines):
                                cv2.putText(
                                    text_img,
                                    line,
                                    (10, 30 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    1,
                                )
                            img = np.concatenate((img, text_img), axis=1)
                        replay_images.append(img)

                    if action_counter == 0:
                        if model.use_cot:
                            goal_tokens = None
                            if teacher_sampler is not None:
                                step_idx = max(t - num_steps_wait, 0)
                                goal_tokens, goal_idx, u = teacher_sampler.get_goal_tokens(step_idx)
                                log_file.write(
                                    f"Teacher goal idx: {goal_idx} (step {step_idx} + {u})\n"
                                )
                                log_file.flush()
                            action, thought = model.step(observation, task_description, goal_tokens=goal_tokens)
                        elif teacher_force_mode == "perspective":
                            action = model.step(
                                observation,
                                task_description,
                                perspective_image=get_libero_camera_image(obs, perspective_obs_key),
                            )
                        else:
                            action = model.step(observation, task_description)
                        action_counter = action.shape[0]

                    step_action = action[-action_counter]
                    obs, reward, done, info = env.step(step_action.tolist())
                    action_counter -= 1
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1
                except Exception as exc:
                    error = str(exc)
                    logger.exception("Episode error")
                    log_file.write(f"Caught exception: {error}\n")
                    traceback.print_exc()
                    break

            task_episodes += 1
            total_episodes += 1

            if debug and replay_images:
                gif_dir = os.path.join(local_log_dir, "videos")
                os.makedirs(gif_dir, exist_ok=True)
                gif_path = os.path.join(
                    gif_dir, f"task{task_id}_episode{episode_idx + 1}.gif"
                )
                save_rollout_gif(replay_images, gif_path, fps=15)

            _, global_episode_idx = episode_is_assigned(
                task_id, episode_idx, num_trials_per_task, rank, world_size
            )
            result = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_idx": episode_idx,
                "global_episode_idx": global_episode_idx,
                "success": bool(done),
                "steps": t,
                "error": error,
            }
            results_file.write(json.dumps(result) + "\n")
            results_file.flush()

            logger.info("Success: %s", done)
            logger.info("# episodes completed so far: %s", total_episodes)
            if total_episodes > 0:
                logger.info(
                    "# successes: %s (%.1f%%)",
                    total_successes,
                    total_successes / total_episodes * 100,
                )
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            if total_episodes > 0:
                log_file.write(
                    f"# successes: {total_successes} "
                    f"({total_successes / total_episodes * 100:.1f}%)\n"
                )
            log_file.flush()

        if task_episodes > 0:
            logger.info(
                "Task %s success rate: %.3f",
                task_id,
                float(task_successes) / float(task_episodes),
            )
            log_file.write(
                f"Task {task_id} success rate: "
                f"{float(task_successes) / float(task_episodes):.3f}\n"
            )
            log_file.flush()
        env.close()
        
    log_file.close()
    results_file.close()

    return total_episodes, total_successes


def parse_args():
    seed_everything(0, workers=True)  # type: ignore
    parser = argparse.ArgumentParser(
        description="Evaluate UniVLA on LIBERO with teacher-forced CoT or perspective images."
    )
    parser.add_argument("--debug", action="store_true", help="Save rollout GIFs.")
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--is_pt_config", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, nargs="+", default="")
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--ckpt_idx", type=int, default=-1)
    parser.add_argument("--emu_hub", type=str, default="")
    parser.add_argument(
        "--vq_hub",
        type=str,
        default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-Base",
    )
    parser.add_argument(
        "--vision_hub",
        type=str,
        default="/share/project/yuqi.wang/OmniSim/pretrain/Emu3-VisionVQ",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
            "libero_spatial_occluded",
            "libero_goal_occluded",
            "libero_10_occluded",
        ],
        required=True,
    )
    parser.add_argument("--device_id", default=0, type=int, help="CUDA device")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug_model", action="store_true")
    parser.add_argument("--no_nccl", action="store_true")
    parser.add_argument("--no_action_ensemble", action="store_true")
    parser.add_argument(
        "--cache_root",
        type=str,
        default="/share/project/yuqi.wang/UniVLA/logs/libero",
        help="Root directory to store cache/logs.",
    )
    parser.add_argument(
        "--with_cot",
        action="store_true",
        help="Enable CoT-style evaluation (subgoal reasoning).",
    )
    parser.add_argument(
        "--teacher_data_path",
        type=str,
        default="data_storage/meta/libero_all_norm.pkl",
        help="Training pickle used to sample teacher-forced subgoal images.",
    )
    parser.add_argument(
        "--teacher_min_h",
        type=int,
        default=5,
        help="Min horizon u for teacher-forced goal index (t + u).",
    )
    parser.add_argument(
        "--teacher_max_h",
        type=int,
        default=10,
        help="Max horizon u for teacher-forced goal index (t + u).",
    )
    parser.add_argument(
        "--teacher_force_mode",
        type=str,
        default="cot",
        choices=["cot", "perspective"],
        help="Use teacher-forced CoT subgoal tokens or GT images from a selected simulator perspective.",
    )
    parser.add_argument(
        "--teacher_image_key",
        type=str,
        default="image",
        help="Dataset image key used when teacher_force_mode=cot.",
    )
    parser.add_argument(
        "--perspective_obs_key",
        type=str,
        default="robot0_eye_in_hand_image",
        help="Observation key used when teacher_force_mode=perspective.",
    )
    parser.add_argument(
        "--no_gripper",
        action="store_true",
        help="Not to use gripper image"
    )
    parser.add_argument(
        "--cot_max_new_tokens",
        type=int,
        default=256,
        help="Max tokens when generating CoT reasoning and goal image.",
    )
    parser.add_argument(
        "--num_trials_per_task", type=int, default=50, help="Episodes per task."
    )
    parser.add_argument(
        "--num_steps_wait", type=int, default=10, help="Initial wait steps."
    )
    parser.add_argument(
        "--camera_resolution",
        type=int,
        default=256,
        help="LIBERO offscreen render resolution per camera.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Optional run id to share across ranks.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_run_id(args):
    if args.run_id:
        return args.run_id
    run_id = f"{args.task_suite_name}-{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    if dist.is_available() and dist.is_initialized():
        obj = [run_id if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(obj, src=0)
        run_id = obj[0]
    return run_id


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)  # type: ignore
    if args.teacher_force_mode == "cot" and not args.with_cot:
        raise ValueError("Teacher-forced evaluation requires --with_cot.")
    if args.teacher_force_mode == "perspective" and args.with_cot:
        raise ValueError("Perspective teacher forcing expects the perspective VLA model, so omit --with_cot.")

    local_rank, rank, world_size = world_info_from_env()

    if not args.no_nccl and world_size > 1:
        local_rank, rank, world_size, render_gpu_device_id = setup_distributed()
    else:
        torch.cuda.set_device(args.device_id)
        render_gpu_device_id = resolve_physical_cuda_device(args.device_id)
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_device_id)

    cache_root = args.cache_root
    os.makedirs(cache_root, exist_ok=True)

    run_id = get_run_id(args)
    eval_log_dir = os.path.join(cache_root, "eval", run_id)
    os.makedirs(eval_log_dir, exist_ok=True)

    if rank == 0:
        meta_path = os.path.join(eval_log_dir, "meta_info.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "task_suite_name": args.task_suite_name,
                    "num_trials_per_task": args.num_trials_per_task,
                    "world_size": world_size,
                    "teacher_force_mode": args.teacher_force_mode,
                    "teacher_data_path": args.teacher_data_path,
                    "teacher_image_key": args.teacher_image_key,
                    "teacher_min_h": args.teacher_min_h,
                    "teacher_max_h": args.teacher_max_h,
                    "perspective_obs_key": args.perspective_obs_key,
                    "results_pattern": "episodes_rank{rank}.jsonl",
                    "log_pattern": "eval_rank{rank}.txt",
                },
                f,
                indent=2,
            )

    if args.teacher_force_mode == "cot" and args.teacher_min_h > args.teacher_max_h:
        raise ValueError(
            f"teacher_min_h ({args.teacher_min_h}) must be <= teacher_max_h ({args.teacher_max_h})"
        )

    teacher_sampler = None
    if args.teacher_force_mode == "cot":
        repo_root = Path(__file__).resolve().parents[4]
        teacher_data_path = args.teacher_data_path
        if not os.path.isabs(teacher_data_path):
            candidate = repo_root / teacher_data_path
            if candidate.exists():
                teacher_data_path = str(candidate)
        if not os.path.exists(teacher_data_path):
            raise FileNotFoundError(f"Teacher data path not found: {teacher_data_path}")

        search_roots = [
            str(Path.cwd()),
            str(repo_root),
            str(Path(teacher_data_path).resolve().parent),
        ]
        teacher_sampler = TeacherForcingSampler(
            data_path=teacher_data_path,
            min_h=args.teacher_min_h,
            max_h=args.teacher_max_h,
            seed=args.seed + rank,
            search_roots=search_roots,
            image_key=args.teacher_image_key,
        )

    model = EmuVLAModel(
        emu_hub=args.emu_hub,
        vq_hub=args.vq_hub,
        vision_hub=args.vision_hub,
        device=torch.device("cuda"),
        use_cot=args.with_cot,
        cot_max_new_tokens=args.cot_max_new_tokens,
        use_gripper= not args.no_gripper
    )

    total_episodes, total_successes = evaluate(
        model=model,
        task_suite_name=args.task_suite_name,
        local_log_dir=eval_log_dir,
        rank=rank,
        world_size=world_size,
        render_gpu_device_id=render_gpu_device_id,
        num_trials_per_task=args.num_trials_per_task,
        num_steps_wait=args.num_steps_wait,
        teacher_sampler=teacher_sampler,
        debug=args.debug,
        teacher_force_mode=args.teacher_force_mode,
        perspective_obs_key=args.perspective_obs_key,
        camera_resolution=args.camera_resolution,
    )

    if dist.is_available() and dist.is_initialized():
        totals = torch.tensor([total_episodes, total_successes], device="cuda")
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        if rank == 0:
            global_episodes = int(totals[0].item())
            global_successes = int(totals[1].item())
            if global_episodes > 0:
                global_sr = global_successes / global_episodes * 100
                logger.info(
                    "Global success rate: %s/%s (%.1f%%)",
                    global_successes,
                    global_episodes,
                    global_sr,
                )
                with open(os.path.join(eval_log_dir, "global_success_rate.txt"), "w") as f:
                    f.write(f"Global success rate: {global_successes}/{global_episodes} ({global_sr}%)")
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    main()
