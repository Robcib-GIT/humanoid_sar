
# ### ESCENARIO   NIST 1                 ######  
### CONFIGURACIÓN NO PRESENTA OBSTACULOS ######

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_entity import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeneratorCfg, TerrainOutput, TerrainGeometry

ALL_TERRAIN_PRESETS: dict[str, Callable[..., SubTerrainCfg]] = {}
_F = TypeVar("_F", bound=Callable[..., SubTerrainCfg])

def terrain_preset(fn: _F) -> _F:
  ALL_TERRAIN_PRESETS[fn.__name__] = fn
  return fn


@dataclass
class StructuredRoomTerrainCfg(SubTerrainCfg):
  wall_height: float = 2.4
  wall_thickness: float = 0.2
  
  # 🎨 COLORES ESTILO UNITY e INDUSTRIALES
  wall_rgba: tuple[float, float, float, float] = (0.35, 0.35, 0.38, 1.0)   
  floor_dark: tuple[float, float, float, float] = (0.20, 0.20, 0.22, 1.0)  
  floor_light: tuple[float, float, float, float] = (0.28, 0.28, 0.30, 1.0) 
  line_rgba: tuple[float, float, float, float] = (0.1, 0.1, 0.1, 1.0)      
  wood_rgba: tuple[float, float, float, float] = (0.65, 0.45, 0.25, 1.0)   
  pallet_rgba: tuple[float, float, float, float] = (0.72, 0.58, 0.42, 1.0) 
  cyl_silver: tuple[float, float, float, float] = (0.9, 0.8, 0.1, 1.0)  

  def function(self, difficulty, spec, rng) -> TerrainOutput:
    geometries = []
    temp_body = spec.worldbody.add_body()

    hx_room, hy_room = 7.0, 7.0
    hx_floor, hy_floor = 12.0, 12.0

    # Piso base negro ampliado
    grid_base = temp_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx_floor, hy_floor, 0.04])
    grid_base.pos = [0.0, 0.0, -0.06]
    geometries.append(TerrainGeometry(geom=grid_base, color=self.line_rgba))

    # Generación de baldosas
    tile_size = 1.0  
    gap = 0.04       
    ht = (tile_size - gap) / 2.0
    for ix in np.arange(-hx_floor + tile_size/2, hx_floor, tile_size):
        for iy in np.arange(-hy_floor + tile_size/2, hy_floor, tile_size):
            color = self.floor_dark if (int(ix) + int(iy)) % 2 == 0 else self.floor_light
            tile = temp_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[ht, ht, 0.05])
            tile.pos = [ix, iy, -0.05]
            geometries.append(TerrainGeometry(geom=tile, color=color))

    # --- FUNCIONES AUXILIARES GEOMÉTRICAS ---
    def add_box(pos, size, rgba, euler=None):
      g = temp_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=size)
      g.pos = pos
      if euler:
        roll, pitch, yaw = euler
        cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
        cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
        cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
        g.quat = [cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy, cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy]
      geometries.append(TerrainGeometry(geom=g, color=rgba))

    def add_cylinder(pos, radius, half_height, rgba):
      g = temp_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radius, half_height])
      g.pos = pos
      geometries.append(TerrainGeometry(geom=g, color=rgba))

    def build_table(cx, cy):
      add_box([cx, cy, 0.75], [0.8, 0.5, 0.02], self.wood_rgba)
      for dx, dy in [(-0.7, -0.4), (0.7, -0.4), (-0.7, 0.4), (0.7, 0.4)]:
        add_box([cx + dx, cy + dy, 0.375], [0.04, 0.04, 0.375], self.wood_rgba)

    def build_pallet(cx, cy, yaw_angle=0.0):
      for dy in [-0.36, 0.0, 0.36]:
        add_box([cx, cy + dy, 0.01], [0.6, 0.04, 0.01], self.pallet_rgba, euler=[0,0,yaw_angle])
      for dx in [-0.55, 0.0, 0.55]:
        for dy in [-0.36, 0.0, 0.36]:
          add_box([cx + dx, cy + dy, 0.05], [0.05, 0.05, 0.03], self.pallet_rgba, euler=[0,0,yaw_angle])
      for dx in [-0.55, -0.27, 0.0, 0.27, 0.55]:
        add_box([cx + dx, cy, 0.09], [0.05, 0.4, 0.01], self.pallet_rgba, euler=[0,0,yaw_angle])

    def build_safety_cone(cx, cy):
      add_box([cx, cy, 0.01], [0.18, 0.18, 0.01], (0.12, 0.12, 0.12, 1.0))
      add_cylinder([cx, cy, 0.20], radius=0.07, half_height=0.19, rgba=(0.95, 0.35, 0.05, 1.0))
      add_cylinder([cx, cy, 0.24], radius=0.071, half_height=0.04, rgba=(0.90, 0.90, 0.95, 1.0))

    # 🛠️ NUEVA FUNCIÓN: RANDOM SPREAD (Dispersión Aleatoria)
    def build_random_spread(center_x, center_y, area_size, num_objects):
      """
      Genera una serie de obstáculos rectangulares aleatorios en una zona determinada.
      """
      for _ in range(num_objects):
        # 1. Calcular una posición aleatoria dentro del área especificada
        dx = rng.uniform(-area_size / 2.0, area_size / 2.0)
        dy = rng.uniform(-area_size / 2.0, area_size / 2.0)
        ox = center_x + dx
        oy = center_y + dy
        
        # 2. Generar tamaños aleatorios para los obstáculos
        # Puedes cambiar estos valores si quieres obstáculos más grandes o pequeños
        size_x = rng.uniform(0.15, 0.5) 
        size_y = rng.uniform(0.15, 0.5)
        size_z = rng.uniform(0.05, 0.3) # Altura del obstáculo
        
        # 3. Rotación aleatoria en el eje Z (Yaw)
        yaw = rng.uniform(-np.pi, np.pi)
        
        # 4. Color aleatorio (tonos grises/industriales para que combine)
        gray_val = rng.uniform(0.4, 0.7)
        color = (gray_val, gray_val, gray_val, 1.0)
        
        # Agregar el obstáculo al entorno (la posición Z es size_z/2 para que repose sobre el suelo)
        add_box([ox, oy, size_z/2.0], [size_x/2.0, size_y/2.0, size_z/2.0], color, euler=[0, 0, yaw])


    # ==========================================
    # CONSTRUCCIÓN ARQUITECTÓNICA 100% CORREGIDA
    # ==========================================
    wt = self.wall_thickness / 2.0
    wh = self.wall_height / 2.0

    # 1. Paredes Exteriores Perimetrales
    add_box([0.0, hy_room, wh], [hx_room, wt, wh], self.wall_rgba)           
    add_box([0.0, -hy_room, wh], [hx_room, wt, wh], self.wall_rgba)          
    add_box([hx_room, 0.0, wh], [wt, hy_room, wh], self.wall_rgba)    
    add_box([-hx_room, 0.0, wh], [wt, hy_room, wh], self.wall_rgba)   

    # 2. EL CENTRO EXACTO (TOTALMENTE ABIERTO)
    add_box([0.0, 0.0, 2.3], [1.5, wt, 0.1], self.wall_rgba) 
    add_box([0.0, 0.0, 2.3], [wt, 1.5, 0.1], self.wall_rgba) 

    # 3. LAS 4 ALAS DE LA CRUZ (CADA UNA CON UNA PUERTA EN EL MEDIO)
    def build_wing_with_door(is_x, sign):
        c1 = 2.525 * sign
        c2 = 5.975 * sign
        h_len = 1.025  
        door_c = 4.25 * sign
        d_len = 0.7    
        
        if is_x:
            add_box([c1, 0.0, wh], [h_len, wt, wh], self.wall_rgba)
            add_box([c2, 0.0, wh], [h_len, wt, wh], self.wall_rgba)
            add_box([door_c, 0.0, 2.3], [d_len, wt, 0.1], self.wall_rgba) 
        else:
            add_box([0.0, c1, wh], [wt, h_len, wh], self.wall_rgba)
            add_box([0.0, c2, wh], [wt, h_len, wh], self.wall_rgba)
            add_box([0.0, door_c, 2.3], [wt, d_len, 0.1], self.wall_rgba) 

    build_wing_with_door(is_x=True, sign=1)   
    build_wing_with_door(is_x=True, sign=-1)  
    build_wing_with_door(is_x=False, sign=1)  
    build_wing_with_door(is_x=False, sign=-1) 

    # ==================================================================================
    # MOBILIARIO Y OBSTÁCULOS
    # ==================================================================================
    
    # --- CUADRANTE 1 (Noroeste): RANDOM SPREAD ---
    # Coordenadas: X negativo, Y positivo.
    # Aquí llamamos a la función que esparce los obstáculos aleatorios.
    # 'area_size=4.0' significa que ocupará un cuadrado de 4x4 metros.
    # 'num_objects=25' es la cantidad de bloques que va a esparcir.
    #build_random_spread(center_x=-4.0, center_y=4.0, area_size=4.5, num_objects=25)

    # Si aún quieres conservar la mesa en una esquina, puedes dejarla así:
    # build_table(cx=-6.0, cy=6.0)

    # --- CUADRANTE 2 (Noreste): Grupo de 3 Cilindros Plateados GRANDES ---
    add_cylinder(pos=[4.0, 4.0, 1.2], radius=0.25, half_height=1.0, rgba=self.cyl_silver)
    add_cylinder(pos=[4.7, 3.5, 1.4], radius=0.25, half_height=1.1, rgba=self.cyl_silver)
    add_cylinder(pos=[3.3, 3.5, 1.0], radius=0.25, half_height=0.9, rgba=self.cyl_silver)

    # --- CUADRANTE 3 (Suroeste): Pallet y Cajas juntas ---
    build_pallet(cx=-4.2, cy=-4.2, yaw_angle=0.0)
    add_box(pos=[-3.2, -4.2, 0.25], size=[0.4, 0.4, 0.25], rgba=self.wood_rgba)
    add_box(pos=[-3.2, -4.2, 0.65], size=[0.2, 0.2, 0.15], rgba=self.wood_rgba, euler=[0, 0, 0.3])
    add_box(pos=[-3.4, -3.3, 0.15], size=[0.3, 0.3, 0.15], rgba=self.wood_rgba, euler=[0, 0, -0.2])

    # --- CUADRANTE 4 (Sureste): Segunda Mesa + CONO DE SEGURIDAD ---
    build_table(cx=3.0, cy=-2.5)
    build_safety_cone(cx=1.8, cy=-2.5)

    return TerrainOutput(origin=np.array([0.0, 0.0, 0.0]), geometries=geometries)


# PRESETS DEL MAPA GLOBAL
@terrain_preset
def structured_room(**overrides: Any) -> StructuredRoomTerrainCfg:
  return StructuredRoomTerrainCfg(**overrides)

@terrain_preset
def flat(**overrides: Any) -> terrain_gen.BoxFlatTerrainCfg:
  return terrain_gen.BoxFlatTerrainCfg(**overrides)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(20.0, 20.0), num_rows=1, num_cols=1,
  sub_terrains={"structured_room": structured_room(proportion=1.0)},
  add_lights=True,
)

THREE_ROOMS_CFG = ROUGH_TERRAINS_CFG

if __name__ == "__main__":
  import mujoco.viewerS
  import torch
  device = "cuda" if torch.cuda.is_available() else "cpu"
  terrain = TerrainEntity(TerrainEntityCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG), device=device)
  print("Cargando escenario con pasajes 100% libres...")
  mujoco.viewer.launch(terrain.spec.compile())

