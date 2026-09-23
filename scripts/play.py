

###### MODO NAVEGACIÓN _ BASADO EN _MAQUINAS DE ESTADOS ######
###### JOSE CARLOS RODRIGUEZ YARAHUAMANA ######
###### PRUEBAS DE NAVEGACIÓN ######
######  TFM ###### 


import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro
import numpy as np

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg	
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value

@dataclass(frozen=True)
class PlayConfig:
    agent: Literal["zero", "random", "trained"] = "trained"
    registry_name: str | None = None
    wandb_run_path: str | None = None
    wandb_checkpoint_name: str | None = None
    checkpoint_file: str | None = None
    motion_file: str | None = None
    num_envs: int | None = None
    device: str | None = None
    video: bool = False
    video_length: int = 200
    video_height: int | None = None
    video_width: int | None = None
    camera: int | str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = False
    log_root: str = "logs/rsl_rl"
    _demo_mode: tyro.conf.Suppress[bool] = False

    # Se eliminan los objetivos por defecto hardcodeados
    target_x: float | None = None
    target_y: float | None = None


# =========================================================================
# 🧠 WRAPPER INTELIGENTE: CONTROL PD + ESCAPE DE GIRO SOBRE EJE + CRONÓMETRO
# =========================================================================
class SmartNavigationPolicy:
    def __init__(self, original_policy, env, target_pos):
        self.original_policy = original_policy
        self.env = env
        self.target_pos = np.array(target_pos)
        
        self.last_dist_error = 0.0
        self.last_yaw_error = 0.0
        self.dt = 0.02 
        
        # Ganancias PD para avance lineal
        self.Kp_lin, self.Kd_lin = 0.6, 0.08
        
        # Ganancias PD optimizadas para un giro rápido
        self.Kp_ang, self.Kd_ang = 5.5, 0.25  

        # Variables para el instinto Anti-Atasco (Giro sobre eje)
        self.step_counter = 0
        self.last_check_dist = 999.0
        self.escape_mode_timer = 0
        self.escape_direction = 1.0 # 1.0 = Izquierda, -1.0 = Derecha

        # Variables de control para parada y reinicio temporizado
        self.goal_reached = False
        self.wait_steps_counter = 0
        self.max_wait_steps = int(10.0 / self.dt) 

        # ⏱️ Cronómetro interno
        self.start_time = _time.time()
        self.trip_duration = 0.0

    def __call__(self, obs):
        self.step_counter += 1

        # 1. Extracción de la odometría de MuJoCo
        mj_data = self.env.unwrapped.sim.data
        try:
            x_val = mj_data.qpos[0].item() if hasattr(mj_data.qpos[0], 'item') else float(mj_data.qpos[0])
            y_val = mj_data.qpos[1].item() if hasattr(mj_data.qpos[1], 'item') else float(mj_data.qpos[1])
            w_val = mj_data.qpos[3].item() if hasattr(mj_data.qpos[3], 'item') else float(mj_data.qpos[3])
            x_q   = mj_data.qpos[4].item() if hasattr(mj_data.qpos[4], 'item') else float(mj_data.qpos[4])
            y_q   = mj_data.qpos[5].item() if hasattr(mj_data.qpos[5], 'item') else float(mj_data.qpos[5])
            z_q   = mj_data.qpos[6].item() if hasattr(mj_data.qpos[6], 'item') else float(mj_data.qpos[6])
        except Exception:
            x_val = mj_data.qpos[0, 0].item() if hasattr(mj_data.qpos, 'item') else float(mj_data.qpos[0, 0])
            y_val = mj_data.qpos[0, 1].item() if hasattr(mj_data.qpos, 'item') else float(mj_data.qpos[0, 1])
            w_val = mj_data.qpos[0, 3].item() if hasattr(mj_data.qpos, 'item') else float(mj_data.qpos[0, 3])
            x_q   = mj_data.qpos[0, 4].item() if hasattr(mj_data.qpos, 'item') else float(mj_data.qpos[0, 4])
            y_q   = mj_data.qpos[0, 5].item() if hasattr(mj_data.qpos[0, 5], 'item') else float(mj_data.qpos[0, 5])
            z_q   = mj_data.qpos[0, 6].item() if hasattr(mj_data.qpos[0, 6], 'item') else float(mj_data.qpos[0, 6])

        current_xy = np.array([x_val, y_val])
        w, x, y, z = w_val, x_q, y_q, z_q

        # 2. Cálculos geométricos
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        current_yaw = np.arctan2(siny_cosp, cosy_cosp)

        delta_vec = self.target_pos - current_xy
        distance = np.linalg.norm(delta_vec)
        target_yaw = np.arctan2(delta_vec[1], delta_vec[0])

        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + np.pi) % (2.0 * np.pi) - np.pi

        # =========================================================================
        # 🛑 MODO ESPERA Y REINICIO AUTOMÁTICO
        # =========================================================================
        if self.goal_reached:
            self.wait_steps_counter += 1
            remaining_time = max(0.0, 10.0 - (self.wait_steps_counter * self.dt))
            
            print(f"\r[⏱️ DURACIÓN DEL VIAJE: {self.trip_duration:.2f}s] | Reiniciando en {remaining_time:4.1f}s...   ", end="", flush=True)
            
            self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, :] = 0.0
            
            if self.wait_steps_counter >= self.max_wait_steps:
                print(f"\n\n[🔄 REINICIO] Tiempo de espera concluido. Regresando al origen...\n", flush=True)
                self.goal_reached = False
                self.wait_steps_counter = 0
                self.step_counter = 0
                self.last_check_dist = 999.0
                self.escape_mode_timer = 0
                self.start_time = _time.time()
                obs, _ = self.env.reset()
            
            return self.original_policy(obs)

        # =========================================================================
        # 🚨 MODO ESCAPE
        # =========================================================================
        if self.escape_mode_timer > 0:
            self.escape_mode_timer -= 1
            cmd_v_x = -0.1  
            cmd_w_z = 2.5 * self.escape_direction      ## ANTES ERA 1.5
            
            print(f"\r[🚨 ATASCADO] Rotando sobre eje hacia {'IZQUIERDA' if self.escape_direction > 0 else 'DERECHA'}... (Quedan {self.escape_mode_timer} ticks)   ", end="", flush=True)
            
            self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, 0] = cmd_v_x
            self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, 1] = 0.0
            self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, 2] = cmd_w_z
            
            self.last_check_dist = distance
            return self.original_policy(obs)

        # =========================================================================
        # 🕵️ MONITOREO DE ATASCOS
        # =========================================================================
        if self.step_counter % 60 == 0:
            if (self.last_check_dist - distance) < 0.03 and distance > 0.5:
                self.escape_mode_timer = 70  
                self.escape_direction = np.random.choice([-1.0, 1.0])
            self.last_check_dist = distance

        # =========================================================================
        # 🟢 MODO NAVEGACIÓN NORMAL (CONTROL PD)
        # =========================================================================
        deriv_dist = (distance - self.last_dist_error) / self.dt
        deriv_yaw = (yaw_error - self.last_yaw_error) / self.dt

        dist_tolerance = 0.15  
        current_elapsed = _time.time() - self.start_time

        if distance > dist_tolerance:
            raw_v_x = self.Kp_lin * distance + self.Kd_lin * deriv_dist
            
            if abs(yaw_error) > 0.2:   
                cmd_v_x = np.clip(raw_v_x, 0.0, 0.15) 
                cmd_v_y = np.clip(0.3 * np.sign(yaw_error), -0.25, 0.25)
            else:
                cmd_v_x = np.clip(raw_v_x, 0.25, 1.0) if raw_v_x > 0 else np.clip(raw_v_x, -1.0, -0.25)
                cmd_v_y = 0.0
                
            cmd_w_z = np.clip(self.Kp_ang * yaw_error + self.Kd_ang * deriv_yaw, -3.0, 3.0)   
            
            print(f"\r[⏱️ {current_elapsed:5.1f}s] Pos: ({current_xy[0]:5.2f}, {current_xy[1]:5.2f}) | Goal: ({self.target_pos[0]:5.2f}, {self.target_pos[1]:5.2f}) | Dist: {distance:4.2f}m   ", end="", flush=True)
        else:
            self.trip_duration = current_elapsed
            cmd_v_x = 0.0
            cmd_v_y = 0.0
            cmd_w_z = 0.0
            self.goal_reached = True
            self.wait_steps_counter = 0
            print(f"\n\n[🏆 ÉXITO] ¡Llegó al Goal! Tiempo total: {self.trip_duration:.2f} segundos. Pos Final: ({current_xy[0]:.2f}, {current_xy[1]:.2f})\n", flush=True)

        self.last_dist_error = distance
        self.last_yaw_error = yaw_error

        self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, 0] = cmd_v_x
        self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, 1] = cmd_v_y
        self.env.unwrapped.command_manager.get_term("twist").vel_command_b[:, 2] = cmd_w_z

        return self.original_policy(obs)


def run_play(task_id: str, cfg: PlayConfig):
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # ─── INTERFAZ DE PROMPT DE COORDENADAS INTERACTIVAS ───
    print("\n" + "="*60)
    print(" 🎯 CONFIGURACIÓN DE OBJETIVO DINÁMICO (ROBOT ESTÁTICO) ")
    print("="*60)
    try:
        chosen_x = float(input("➡️  Ingresa la coordenada Target X (ejemplo: -7.94): "))
        chosen_y = float(input("➡️  Ingresa la coordenada Target Y (ejemplo: -4.71): "))
    except ValueError:
        print("\n❌ Error: Debes ingresar números válidos. Usando valores de emergencia [-7.94, -4.71]")
        chosen_x, chosen_y = -7.94, -4.71
    print("="*60 + "\n")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    DUMMY_MODE = cfg.agent in {"zero", "random"}
    TRAINED_MODE = not DUMMY_MODE

    if cfg.no_terminations:
        env_cfg.terminations = {}

    is_tracking_task = "motion" in env_cfg.commands and isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    if is_tracking_task and cfg._demo_mode:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        motion_cmd.sampling_mode = "uniform"

    if is_tracking_task:
        motion_cmd = env_cfg.commands["motion"]
        assert isinstance(motion_cmd, MotionCommandCfg)
        if cfg.motion_file is not None and Path(cfg.motion_file).exists():
            motion_cmd.motion_file = cfg.motion_file
        elif DUMMY_MODE:
            if not cfg.registry_name: raise ValueError("Error en motion artifacts.")
            registry_name = cfg.registry_name
            if ":" not in registry_name: registry_name += ":latest"
            import wandb
            api = wandb.Api()
            artifact = api.artifact(registry_name)
            motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
        else:
            if cfg.motion_file is not None:
                motion_cmd.motion_file = cfg.motion_file
            else:
                import wandb
                api = wandb.Api()
                if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
                    raise ValueError("Falta wandb_run_path.")
                if cfg.wandb_run_path is not None:
                    wandb_run = api.run(str(cfg.wandb_run_path))
                    art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
                    if art is None: raise RuntimeError("No motion artifact found.")
                    motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if TRAINED_MODE:
        log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
        if cfg.checkpoint_file is not None:
            resume_path = Path(cfg.checkpoint_file)
            if not resume_path.exists(): raise FileNotFoundError(f"Checkpoint no encontrado: {resume_path}")
        else:
            if cfg.wandb_run_path is None: raise ValueError("wandb_run_path requerido.")
            resume_path, was_cached = get_wandb_checkpoint_path(log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name)
        log_dir = resume_path.parent

    if cfg.num_envs is not None: env_cfg.scene.num_envs = cfg.num_envs
    else: env_cfg.scene.num_envs = 1

    if cfg.video_height is not None: env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None: env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if TRAINED_MODE and cfg.video:
        assert log_dir is not None
        env = VideoRecorder(env, video_folder=log_dir / "videos" / "play", step_trigger=lambda step: step == 0, video_length=cfg.video_length, disable_logger=True)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if DUMMY_MODE:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if cfg.agent == "zero":
            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor: return torch.zeros(action_shape, device=env.unwrapped.device)
            policy = PolicyZero()
        else:
            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor: return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
            policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device)
        policy = runner.get_inference_policy(device=device)

    ckpt_manager: CheckpointManager | None = None
    if TRAINED_MODE and resume_path is not None:
        _ckpt_runner = runner
        def _reload_policy(path: str):
            _ckpt_runner.load(path, load_cfg={"actor": True}, strict=True, map_location=device)
            return _ckpt_runner.get_inference_policy(device=device)

        if cfg.wandb_run_path is None:
            ckpt_dir = resume_path.parent
            def fetch_available_local() -> list[tuple[str, str]]:
                now = _time.time()
                entries: list[tuple[str, str, int]] = []
                for f in sorted(ckpt_dir.glob("*.pt")):
                    try: step = int(f.stem.split("_")[1])
                    except (IndexError, ValueError): step = 0
                    ago = format_time_ago(int(now - f.stat().st_mtime))
                    entries.append((f.name, ago, step))
                entries.sort(key=lambda x: x[2])
                return [(name, t) for name, t, _ in entries]
            ckpt_manager = CheckpointManager(current_name=resume_path.name, fetch_available=fetch_available_local, load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)))

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
        del has_display
    else:
        resolved_viewer = cfg.viewer

    # =========================================================================
    # 🔗 ACTIVACIÓN DEL WRAPPER INTELIGENTE (CON COORDENADAS INGRESADAS)
    # =========================================================================
    print(f"\n[CONTROL]: Iniciando Navegación Autónoma hacia el Goal ingresado: [{chosen_x}, {chosen_y}]\n")
    policy = SmartNavigationPolicy(policy, env, target_pos=[chosen_x, chosen_y])

    if resolved_viewer == "native": NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser": ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
    else: raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()

def main():
    maybe_print_top_level_help("play")
    import mjlab.tasks  # noqa: F401
    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(tyro.extras.literal_type_from_choices(all_tasks), add_help=False, return_unknown_args=True, config=mjlab.TYRO_FLAGS)
    agent_cfg = load_rl_cfg(chosen_task)
    args = tyro.cli(PlayConfig, args=remaining_args, default=PlayConfig(), prog=sys.argv[0] + f" {chosen_task}", config=mjlab.TYRO_FLAGS)
    del remaining_args, agent_cfg
    run_play(chosen_task, args)

if __name__ == "__main__":
    main()



