

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