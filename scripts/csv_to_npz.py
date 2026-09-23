######  CONVERSIÓN  DATA  CSV A NPZ  ######
######  JOSE CARLOS RODRIGUEZ YARAHUAMAN ###### 
 

from typing import Any
import numpy as np
import torch
import tyro
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device | str,
        line_range: tuple[int, int] | None = None,
        z_offset: float = 0.0,
        swap_yz: bool = False,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.line_range = line_range
        self.z_offset = z_offset
        self.swap_yz = swap_yz

        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Carga la moción adaptada al CSV con corrección de altura y ejes."""
        raw_data = np.loadtxt(self.motion_file, delimiter=",", skiprows=1)

        if self.line_range is not None:
            start, end = self.line_range
            raw_data = raw_data[start:end]

        motion = torch.from_numpy(raw_data).to(torch.float32).to(self.device)

        # 1. POSICIÓN (Columnas 1, 2, 3 en cm -> m)
        if self.swap_yz:
            # En caso de exportaciones donde Y es vertical (Up) y Z profundidad
            pos_x = motion[:, 1:2] / 100.0
            pos_y = motion[:, 3:4] / 100.0
            pos_z = (motion[:, 2:3] / 100.0) - self.z_offset
            self.motion_base_poss_input = torch.cat([pos_x, pos_y, pos_z], dim=1)
        else:
            # Estándar MuJoCo: Z es la altura vertical
            self.motion_base_poss_input = motion[:, 1:4] / 100.0
            self.motion_base_poss_input[:, 2] -= self.z_offset

        print(f"[INFO] Altura inicial Z de la pelvis: {self.motion_base_poss_input[0, 2]:.4f} m")

        # 2. ROTACIÓN (Columnas 4, 5, 6): Euler XYZ (grados) -> Quaternion WXYZ
        euler_angles_deg = raw_data[:, 4:7]
        rot = R.from_euler("xyz", euler_angles_deg, degrees=True)
        quat_xyzw = rot.as_quat()

        # Reordenamos de XYZW a WXYZ para MuJoCo
        quat_wxyz = np.zeros_like(quat_xyzw)
        quat_wxyz[:, 0] = quat_xyzw[:, 3]
        quat_wxyz[:, 1:] = quat_xyzw[:, :3]
        self.motion_base_rots_input = torch.from_numpy(quat_wxyz).to(torch.float32).to(self.device)

        # 3. ARTICULACIONES (Columna 7 en adelante)
        self.motion_dof_poss_input = torch.deg2rad(motion[:, 7:])

        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(
            0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
        )
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(
            self.motion_base_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_dof_vels = torch.gradient(
            self.motion_dof_poss, spacing=self.output_dt, dim=0
        )[0]
        self.motion_base_ang_vels = self._so3_derivative(
            self.motion_base_rots, self.output_dt
        )

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        return omega

    def get_next_state(self) -> tuple[tuple[torch.Tensor, ...], bool]:
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_sim(
    sim: Simulation,
    scene: Scene,
    joint_names: list[str],
    input_file: str,
    input_fps: float,
    output_fps: float,
    output_name: str,
    render: bool,
    line_range: tuple[int, int] | None,
    z_offset: float = 0.0,
    swap_yz: bool = False,
    renderer: OffscreenRenderer | None = None,
):
    motion = MotionLoader(
        motion_file=input_file,
        input_fps=int(input_fps),
        output_fps=int(output_fps),
        device=sim.device,
        line_range=line_range,
        z_offset=z_offset,
        swap_yz=swap_yz,
    )

    robot: Entity = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    log: dict[str, Any] = {
        "fps": [output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    frames = []
    scene.reset()

    print(f"\nStarting simulation with {motion.output_frames} frames...")
    pbar = tqdm(total=motion.output_frames, desc="Processing frames", unit="frame", ncols=100)

    while not file_saved:
        (state, reset_flag) = motion.get_next_state()
        (m_pos, m_rot, m_lvel, m_avel, m_dpos, m_dvel) = state

        root_states = robot.data.default_root_state.clone()
        root_states[:, 0:3] = m_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = m_rot
        root_states[:, 7:10] = m_lvel
        root_states[:, 10:] = m_avel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = m_dpos
        joint_vel[:, robot_joint_indexes] = m_dvel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.forward()
        scene.update(sim.mj_model.opt.timestep)

        if render and renderer is not None:
            renderer.update(sim.data)
            frames.append(renderer.render())

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy())

            pbar.update(1)

            if reset_flag:
                file_saved = True
                pbar.close()

                print("\nSaving data to /tmp/motion.npz...")
                for k in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
                    log[k] = np.stack(log[k], axis=0)
                np.savez("/tmp/motion.npz", **log)

                import wandb
                run = wandb.init(project="csv_to_npz", name=output_name)
                run.log_artifact("/tmp/motion.npz", name=output_name, type="motions")

                if render:
                    import mediapy as media
                    print("Creating video...")
                    media.write_video("./motion.mp4", frames, fps=output_fps)
                    wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})
                wandb.finish()


def main(
    input_file: str,
    output_name: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0,
    device: str = "cuda:0",
    render: bool = False,
    line_range: tuple[int, int] | None = None,
    z_offset: float = 0.0,
    swap_yz: bool = False,
):
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARNING]: CUDA not available. Falling back to CPU.")
        device = "cpu"

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / output_fps
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
    model = scene.compile()
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)

    renderer = None
    if render:
        viewer_cfg = ViewerConfig(height=640, width=1280, entity_name="robot", distance=4.5)
        renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
        renderer.initialize()

    run_sim(
        sim=sim,
        scene=scene,
        joint_names=[
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
            "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ],
        input_fps=input_fps,
        input_file=input_file,
        output_fps=output_fps,
        output_name=output_name,
        render=render,
        line_range=line_range,
        z_offset=z_offset,
        swap_yz=swap_yz,
        renderer=renderer,
    )


if __name__ == "__main__":
    tyro.cli(main, config=mjlab.TYRO_FLAGS)

















