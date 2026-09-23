

# humanoid_sar: Locomoción y Control de un Robot Humanoide en Escenarios SAR

> Locomocion y Manipulación Diestra Bimanual de Un Robot Humanoide para Tareas de Busqueda y Rescate.

---

## Características

- **Modelado y control dinámico:** Simulación física de alta fidelidad del robot humanoide Unitree G1 sobre MuJoCo.
- **Aceleración por GPU:** Integración nativa con el framework `mjlab` para entornos paralelizados.
- **Pipelines DRL:** Entrenamiento, optimización (PPO) y evaluación de políticas dinámicas de locomoción.
- **Navegación reactiva:** Control FSM-PD adaptado para franqueo de obstáculos en arenas NIST (códigos amarillo y naranja).

---

## Requisitos previos

- **Sistema operativo:** Linux (probado en Ubuntu 22.04 LTS).
- **GPU:** Tarjeta gráfica compatible con NVIDIA CUDA.
- **Gestor de paquetes:** [uv](https://docs.astral.sh/uv/) (recomendado para aislamiento y reproducibilidad ultrarrápida):
  bash
  curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh




## 🎮 Ejecución y Simulación

Todos los comandos se gestionan mediante `uv` para garantizar la ejecución reproducible del entorno en GPU.

---

### 1. Locomoción en terreno irregular con escombros (`play`)

Para visualizar y evaluar la política de locomoción dinámica y franqueo de obstáculos del Unitree G1 sobre terreno irregular (*Rough Terrain*):

:```bash
uv run play Mjlab-Velocity-Rough-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_velocity/2026-07-24_14-11-23/model_52400.pt \
    --device "cuda:0"
:```bash

# 2 Seguimiento Cinemático y Manipulación Bimanual (`Tracking Motion`)

> Módulo de ejecución y evaluación de trayectorias cinemáticas de referencia para el robot humanoide **Unitree G1**, enfocado en tareas de asistencia, contacto controlado y manipulación bimanual en entornos de búsqueda y rescate.

---

## 📌 Descripción general

Este módulo permite reproducir y evaluar políticas de control entrenadas para el seguimiento preciso de trayectorias articulares generadas previamente (mediante retargeting o síntesis de movimiento). El objetivo principal es dotar al humanoide de movimientos coordinados en torso y tren superior para interactuar de forma segura con el entorno o asistir a víctimas.

---

## ⚙️ Requisitos previos

- Archivo de trayectoria cinemática de referencia en formato `.npz` (posiciones, orientaciones y velocidades articulares).
- Checkpoint del modelo entrenado (`.pt`) correspondiente a la tarea de *tracking*.
- Dependencias del entorno activadas mediante `uv`.

---

## 🚀 Comandos de ejecución

### Evaluación y visualización directa (`play`)

Para cargar la política entrenada y reproducir la trayectoria de referencia (`movimiento_cura.npz`) en simulación:

<img width="1747" height="855" alt="RCP_NEW" src="https://github.com/user-attachments/assets/4321d55b-9a97-422c-b40a-f044ea484ebc" />


```bash
uv run play Mjlab-Tracking-Flat-Unitree-G1 \
    --checkpoint-file logs/rsl_rl/g1_tracking/2026-09-09_14-46-49/model_124500.pt \
    --motion-file "movimiento_cura.npz" \
    --device "cuda:0"
  


