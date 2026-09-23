# ###########################################################
# ##################### TERRENO NUMERO 2 #################### 
# ######## JOSE CARLOS RODRIGUEZ YARAHUAMAAN ################

    

#ULTIMO DESARROLLO FINALIZADO - AMBIENTE NUMERO 2 ###



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


# 1. CONFIGURACIÓN DEL PRESET NATIVO DE MJLAB (Para el Cuadrante 1)
@terrain_preset
def random_spread_boxes(
  **overrides: Any,
) -> terrain_gen.BoxRandomSpreadTerrainCfg:
  defaults: dict[str, Any] = dict(
    num_boxes=165,                 
    box_width_range=(0.1, 0.8),   
    box_length_range=(0.1, 1.5),
    box_height_range=(0.05, 0.3),
    platform_width=1.0,
    border_width=0.25,
    size=(5.8, 5.8),              
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomSpreadTerrainCfg(**defaults)

# 2. CONFIGURACIÓN DEL PRESET NATIVO RANDOM GRID (Para el Cuadrante 2)
@terrain_preset
def native_random_grid(
  **overrides: Any,
) -> terrain_gen.BoxRandomGridTerrainCfg:
  defaults: dict[str, Any] = dict(
    size=(6.7, 6.7),
    grid_width=0.4,                  
    grid_height_range=(0.06, 0.50),  
  )
  defaults.update(overrides)
  return terrain_gen.BoxRandomGridTerrainCfg(**defaults)

# 3. CONFIGURACIÓN DEL PRESET NATIVO RANDOM UNIFORM (Para el Cuadrante 4)
@terrain_preset
def native_random_uniform(
  **overrides: Any,
) -> terrain_gen.HfRandomUniformTerrainCfg:
  defaults: dict[str, Any] = dict(
    size=(5.8, 5.8),
    noise_range=(-0.1, 0.1),         
    downsampled_scale=0.2,           
  )
  defaults.update(overrides)
  return terrain_gen.HfRandomUniformTerrainCfg(**defaults)


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
  cyl_silver: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)  

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

    # --- FUNCION AUXILIAR GEOMÉTRICA CENTRALIZADA ---
    

    # ==========================================
    # 📦 NUEVOS OBJETOS: CAJAS, RAMPAS Y CONOS
    # ==========================================

    def build_slatted_crate(cx, cy, cz=0.0):
        """Genera una caja de madera con listones separados (Hueca por dentro)."""
        c_wood = (0.40, 0.26, 0.13, 1.0)
        hw, hl, hh = 0.3, 0.2, 0.25  # Mitad de ancho, largo y alto
        
        # Base
        add_box([cx, cy, cz + 0.02], [hw, hl, 0.02], c_wood)
        # 4 Postes en las esquinas
        for dx in [-hw+0.02, hw-0.02]:
            for dy in [-hl+0.02, hl-0.02]:
                add_box([cx + dx, cy + dy, cz + hh], [0.02, 0.02, hh], c_wood)
        # Listones laterales (3 niveles de altura)
        for hz in [0.08, 0.25, 0.42]: 
            for dy in [-hl+0.02, hl-0.02]: # Lados largos
                add_box([cx, cy + dy, cz + hz], [hw, 0.01, 0.04], c_wood)
            for dx in [-hw+0.02, hw-0.02]: # Lados cortos
                add_box([cx + dx, cy, cz + hz], [0.01, hl-0.04, 0.04], c_wood)

    def build_solid_crate(cx, cy, cz=0.0):
        """Genera una caja de envío sólida y cerrada con marcos reforzados."""
        c_wood_light = (0.85, 0.72, 0.55, 1.0)
        c_wood_dark = (0.75, 0.60, 0.40, 1.0)
        # Cuerpo principal (claro)
        add_box([cx, cy, cz + 0.3], [0.25, 0.4, 0.3], c_wood_light)
        # Refuerzos horizontales (oscuros)
        add_box([cx, cy, cz + 0.58], [0.26, 0.41, 0.02], c_wood_dark)
        add_box([cx, cy, cz + 0.02], [0.26, 0.41, 0.02], c_wood_dark)

    def build_agility_ramp(cx, cy):
        """Genera dos rampas de madera inclinadas estilo agility canino."""
        c_wood = (0.65, 0.46, 0.28, 1.0)
        # Rampa 1 (Sube)
        add_box([cx - 0.8, cy + 0.3, 0.15], [0.6, 0.25, 0.03], c_wood, euler=[0, -0.25, 0])
        # Rampa 2 (Baja - ligeramente desfasada como en la foto)
        add_box([cx + 0.8, cy - 0.3, 0.15], [0.6, 0.25, 0.03], c_wood, euler=[0, 0.25, 0])
        # Conexión central plana
        add_box([cx, cy, 0.30], [0.2, 0.55, 0.03], c_wood)

    def build_traffic_cone(cx, cy):
        """Genera un cono de señalización rojo y gris mediante cilindros apilados."""
        c_red = (0.8, 0.25, 0.25, 1.0)
        c_grey = (0.7, 0.7, 0.7, 1.0)
        # Base gruesa roja
        add_cylinder([cx, cy, 0.05], radius=0.08, half_height=0.05, rgba=c_red)
        # Banda central gris
        add_cylinder([cx, cy, 0.15], radius=0.06, half_height=0.05, rgba=c_grey)
        # Punta roja
        add_cylinder([cx, cy, 0.25], radius=0.04, half_height=0.05, rgba=c_red)
     


# 🚧 OBSTÁCULO INDUSTRIAL: ROMPEMUELLES EN FRANJAS AMARILLAS Y NEGRAS
# 🚧 ROMPEMUELLES PEQUEÑO Y REDONDEADO (Estilo residencial/estacionamiento)
    def build_speed_bump(cx, cy, yaw_angle=0.0):
        """Genera un rompemuelles pequeño y curvo usando secciones de cilindros alineados."""
        c_yellow = (0.95, 0.75, 0.05, 1.0) # Amarillo tráfico
        c_black = (0.15, 0.15, 0.16, 1.0)  # Negro goma

        # Parámetros para que sea pequeño como en la foto
        num_modulos = 8        # Cantidad de bloques alternados
        ancho_modulo = 0.18    # Ancho de cada franja (X) -> Más angosto
        largo_total = 1.2      # Largo del reductor (Y) -> Más corto, ideal para el paso del G1
        
        # Parámetros para la curva superior (Perfil curvo sutil)
        radio_cilindro = 0.25   # Radio grande para que la curva superior sea muy suave
        alto_visible = 0.04     # Altura real que sobresale del suelo (4 cm, ideal para el robot)

        # Calculamos el inicio y desfase para centrarlo en (cx, cy)
        start_x = cx - (num_modulos * ancho_modulo) / 2.0
        half_w = ancho_modulo / 2.0
        half_l = largo_total / 2.0

        # Posición de entierro para que solo sobresalga el 'alto_visible'
        z_pos = alto_visible - radio_cilindro 

        for i in range(num_modulos):
            # Alternamos color
            color = c_black if i % 2 == 0 else c_yellow
            
            # Cálculo de la posición X local
            x_local = start_x + i * ancho_modulo + half_w
            
            # Aplicar rotación (yaw) si se desea colocar en diagonal
            dx = (x_local - cx) * np.cos(yaw_angle)
            dy = (x_local - cx) * np.sin(yaw_angle)
            
            # En MuJoCo, para acostar un cilindro horizontalmente a lo largo del eje Y, 
            # lo rotamos en el eje X (Roll = 90 grados -> pi/2)
            roll = np.pi / 2.0
            pitch = 0.0
            yaw = yaw_angle

            # Calcular cuaternión combinando las rotaciones
            cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
            cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
            cy_w, sy_w = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
            
            quat = [
                cr*cp*cy_w + sr*sp*sy_w,
                sr*cp*cy_w - cr*sp*sy_w,
                cr*sp*cy_w + sr*cp*sy_w,
                cr*cp*sy_w - sr*sp*cy_w
            ]
            # Añadimos el módulo cilíndrico (size=[radio, mitad_de_longitud])
            g = temp_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[radio_cilindro, half_l])
            g.pos = [cx + dx, cy + dy, z_pos]
            g.quat = quat
            geometries.append(TerrainGeometry(geom=g, color=color))

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

# 🚧 1. ESCALERA PIRAMIDAL LARGA Y ESQUINADA (DIAGONAL)
    def build_single_long_stairs(cx, cy, yaw_angle=np.pi / 4):
        """Genera UNA SOLA escalera piramidal alargada rotada en diagonal (esquinada).
        
        yaw_angle = np.pi / 4 equivale a 45 grados de rotación.
        """
        w = 1.4 / 2.0     # Mitad del ancho total (1.4m totales)
        d = 0.2 / 1.0     # Profundidad del escalón
        c_stair = (0.40, 0.26, 0.13, 1.0) 

        def add_rotated_step(offset_x, offset_z):
            """Calcula la posición de cada escalón aplicando la matriz de rotación en el plano XY."""
            # Transformación de coordenadas para mantener los escalones alineados en el sentido de la marcha
            dx = offset_x * np.cos(yaw_angle)
            dy = offset_x * np.sin(yaw_angle)
            
            pos_final = [cx + dx, cy + dy, offset_z]
            # Pasamos yaw_angle en el tercer componente de euler para rotar la caja
            add_box(pos_final, [d, w, offset_z], c_stair, euler=[0, 0, yaw_angle])

        # --- CONSTRUCCIÓN DE ESCALONES ROTADOS ---
        # Lado izquierdo (Subida)
        add_rotated_step(-0.9, 0.075) 
        add_rotated_step(-0.6, 0.150) 
        add_rotated_step(-0.3, 0.225) 
        
        # Lado derecho (Bajada)
        add_rotated_step(0.3, 0.225) 
        add_rotated_step(0.6, 0.150) 
        add_rotated_step(0.9, 0.075) 

        # Plataforma superior unificada central
        add_rotated_step(0.0, 0.30)


# 🤸 OBSTÁCULO DE GIMNASIA CORREGIDO (Rampa continua hasta el suelo)
    def build_gym_step_ramp(cx, cy, yaw_angle=0.0):
        """Genera el módulo de gimnasia con 3 escalones y rampa limpia que muere en el suelo."""
        c_blue = (0.40, 0.26, 0.13, 1.0) # Azul colchoneta uniforme
        w = 1.3 / 2.0  # Mitad del ancho (1.3 metros de ancho total)

        def add_rotated_part(offset_x, offset_z, size, pitch=0.0):
            """Calcula la transformación de posición exacta evitando desfaces."""
            dx = offset_x * np.cos(yaw_angle)
            dy = offset_x * np.sin(yaw_angle)
            pos_final = [cx + dx, cy + dy, offset_z]
            add_box(pos_final, size, c_blue, euler=[0, pitch, yaw_angle])

        # --- 1. LADO DE ESCALONES (Estructura de subida maciza) ---
        # Escalón 1 (Base inferior: 15 cm de alto)
        add_rotated_part(-0.75, 0.075, [0.15, w, 0.075])
        
        # Escalón 2 (Intermedio: 30 cm de alto)
        add_rotated_part(-0.45, 0.150, [0.15, w, 0.150])
        
        # Escalón 3 / Meseta plana superior (45 cm de alto)
        add_rotated_part(-0.15, 0.225, [0.15, w, 0.225])

        # --- 2. LADO DE RAMPA (Caída limpia hasta Z = 0) ---
        # Longitud en X = 1.0m, Alto = 0.45m. 
        # Hipotenusa = 1.096m -> half_extent = 0.548m
        # Ángulo pitch exacto = 24.22 grados -> 0.4228 rad
        add_rotated_part(0.5, 0.225, [0.548, w, 0.02], pitch=0.4228)
        
        # Soporte interno en cuña (Oculto debajo de la rampa para rellenar sin romper la pendiente)
        add_rotated_part(0.3, 0.1125, [0.3, w, 0.1125])






    # 📦 PISTA REALISTA: PALLET INDUSTRIAL CARGADO CON CAJAS
    def build_loaded_pallet(cx, cy, yaw_angle=0.0):
        """Genera un pallet de madera detallado con un bloque de cajas de cartón encima."""
        c_wood = (0.78, 0.64, 0.48, 1.0)   
        c_carton = (0.74, 0.56, 0.38, 1.0) 

        # --- Estructura del Pallet ---
        for dy in [-0.36, 0.0, 0.36]:
            add_box([cx, cy + dy, 0.01], [0.6, 0.04, 0.01], c_wood, euler=[0, 0, yaw_angle])
        for dx in [-0.55, 0.0, 0.55]:
            for dy in [-0.36, 0.0, 0.36]:
                add_box([cx + dx, cy + dy, 0.05], [0.05, 0.05, 0.03], c_wood, euler=[0, 0, yaw_angle])
        for dx in [-0.55, -0.27, 0.0, 0.27, 0.55]:
            add_box([cx + dx, cy, 0.09], [0.05, 0.4, 0.01], c_wood, euler=[0, 0, yaw_angle])

        # --- Cajas de Cartón Apiladas ---
        box_w, box_l, box_h = 0.25, 0.35, 0.20  
        z_base = 0.10

        add_box([cx - 0.26, cy - 0.02, z_base + box_h], [box_w, box_l, box_h], c_carton, euler=[0, 0, yaw_angle])
        add_box([cx + 0.26, cy - 0.02, z_base + box_h], [box_w, box_l, box_h], c_carton, euler=[0, 0, yaw_angle])
        add_box([cx - 0.26, cy - 0.02, z_base + 3*box_h], [box_w, box_l, box_h], c_carton, euler=[0, 0, yaw_angle])
        add_box([cx + 0.26, cy - 0.02, z_base + 3*box_h], [box_w, box_l, box_h], c_carton, euler=[0, 0, yaw_angle])

    # 🚧 NUEVA FUNCIÓN OPTIMIZADA: CUADRÍCULA VERDE EN CUADRADOS SEPARADOS
    def build_separated_green_grid(cx, cy):
        """Genera cuadrados verdes individuales con alturas aleatorias y separación visible."""
        step_w = 0.4      
        gap = 0.04        
        half_w = (step_w - gap) / 2.0  
        
        grid_size = 14    
        start_x = cx - (grid_size * step_w) / 2.0
        start_y = cy - (grid_size * step_w) / 2.0
        
        for i in range(grid_size):
            for j in range(grid_size):
                h = float(rng.uniform(0.05, 0.40)) 
                gx = start_x + i * step_w + (step_w / 2.0)
                gy = start_y + j * step_w + (step_w / 2.0)
                
                color_verde = (0.2, 0.6, 0.3, 1.0) 
                add_box([gx, gy, h/2.0 + 0.05], [half_w, half_w, h/2.0], color_verde)

    # ==========================================
    # CONSTRUCCIÓN ARQUITECTÓNICA
    # ==========================================
    wt = self.wall_thickness / 2.0
    wh = self.wall_height / 2.0

    add_box([0.0, hy_room, wh], [hx_room, wt, wh], self.wall_rgba)           
    add_box([0.0, -hy_room, wh], [hx_room, wt, wh], self.wall_rgba)          
    add_box([hx_room, 0.0, wh], [wt, hy_room, wh], self.wall_rgba)    
    add_box([-hx_room, 0.0, wh], [wt, hy_room, wh], self.wall_rgba)   

    add_box([0.0, 0.0, 2.3], [1.5, wt, 0.1], self.wall_rgba) 
    add_box([0.0, 0.0, 2.3], [wt, 1.5, 0.1], self.wall_rgba) 

    def build_wing_with_door(is_x, sign):
        c1, c2, h_len, door_c, d_len = 2.525 * sign, 5.975 * sign, 1.025, 4.25 * sign, 0.7    
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
    # DISTRIBUCIÓN POR CUADRANTES
    # ==================================================================================
    
    # --- CUADRANTE 1 (Noroeste): INTACTO (Tablas de madera y cilindros) ---
    cfg_spread = random_spread_boxes()
    output_spread = cfg_spread.function(difficulty, spec, rng)
    for geom_obj in output_spread.geometries:
        if geom_obj.geom.size[0] >= 1.8 or geom_obj.geom.size[1] >= 1.8:
            continue
            
        color_realista = (0.95, 0.40, 0.05, 1.0)
        # 🎯 CORRECCIÓN: Eliminamos las líneas problemáticas de material=None
        geom_obj.color = color_realista
        geom_obj.geom.rgba = color_realista
        
        geom_obj.geom.pos = [
            geom_obj.geom.pos[0] - 6.5, 
            geom_obj.geom.pos[1] + 0.5, 
            geom_obj.geom.pos[2]
        ]
        geometries.append(geom_obj)
        
    add_cylinder(pos=[-6.0, 5.8, 0.9], radius=0.20, half_height=1.5, rgba=self.cyl_silver)
    add_cylinder(pos=[-5.3, 6.2, 0.9], radius=0.25, half_height=2.1, rgba=self.cyl_silver)
    add_cylinder(pos=[-3.7, 3.5, 0.9], radius=0.25, half_height=1.9, rgba=self.cyl_silver)


    # --- CUADRANTE 2 (Noreste): RANDOM GRID AMARILLO ---
    cfg_random = native_random_grid()
    output_random = cfg_random.function(difficulty, spec, rng)
    for geom_obj in output_random.geometries:
        color_amarillo = (0.95, 0.85, 0.15, 1.0)
        geom_obj.color = color_amarillo
        geom_obj.geom.rgba = color_amarillo
        
        geom_obj.geom.pos = [
            geom_obj.geom.pos[0] - 0.10, 
            geom_obj.geom.pos[1] + 0.38, 
            geom_obj.geom.pos[2] + 0.13,
        ]
        geometries.append(geom_obj)


    # --- CUADRANTE 3 (Suroeste): REJILLA DE ESCALERAS ---

   # --- CUADRANTE 3 (Suroeste): ZONA DE ESCALERAS SIMPLIFICADA ---
    
# --- CUADRANTE 3 (Suroeste): ZONA DE PISADA Y LOCULACIÓN ---
# --- CUADRANTE 3 (Suroeste): ZONA DE ESCALERAS Y GIMNASIA ---
    
    # 1. Tu escalera gris gigante y alargada en el centro del cuarto
    build_single_long_stairs(cx=-4.5, cy=-4.5 , yaw_angle=114.0)
    add_cylinder(pos=[-6.7, -1.5, 0.9], radius=0.25, half_height=1.9, rgba=self.cyl_silver)
    add_cylinder(pos=[-6.2, -1.1, 0.9], radius=0.25, half_height=2.9, rgba=self.cyl_silver)
    add_cylinder(pos=[-6.5, -3.9, 0.9], radius=0.25, half_height=1.9, rgba=self.cyl_silver)

    # 2. Tu nuevo objeto de gimnasia azul corregido y macizo
    # Lo ubicamos en X=-1.5 (cerca del pasillo divisor) y Y=-3.5 para que esté perfectamente alineado con la escalera gris.
    build_gym_step_ramp(cx=-4.5, cy=-1.5, yaw_angle=114.7)
    build_gym_step_ramp(cx=-1.5, cy=-4.3, yaw_angle=0.0)
    build_gym_step_ramp(cx=-1.9, cy=-1.8, yaw_angle=45.0)
    # --- CUADRANTE 4 (Sureste): CUADRÍCULA VERDE EN CUADRADOS SEPARADOS ---
    # Centrado perfectamente en tu cuarta sección
   # build_separated_green_grid(cx=4.0, cy=-4.0)
     
    # Pallet de madera cargado con cajas estilo la imagen
    #build_loaded_pallet(cx=-6.5, cy=0.5, yaw_angle=0.0)
     # --- CUADRANTE 1 (Noroeste): ALMACÉN DE PRUEBAS ---
    # ... (tus líneas anteriores de output_spread) ...
    
    # 📦 Colocamos algunas cajas de listones y sólidas esparcidas
    build_slatted_crate(cx=3.2, cy=-4.5)
    #build_slatted_crate(cx=2.2, cy=-2.0)
    build_solid_crate(cx=4.5, cy=-2.7)

    # 🚧 Colocamos la rampa de Agility en un espacio libre
    build_agility_ramp(cx=3.2, cy=-3.5)
    #build_agility_ramp(cx=6.2, cy=-5.5)
    # 🔺 Añadimos los conos de tráfico alrededor de la rampa
    build_traffic_cone(cx=-6.3, cy=4.0)
    build_traffic_cone(cx=-6.4, cy=3.0)
    build_traffic_cone(cx=-6.5, cy=4.0)
    build_speed_bump(cx=2.25, cy=-1.25, yaw_angle=0.0)
    build_speed_bump(cx=4.25, cy=-1.35, yaw_angle=0.0)
    build_speed_bump(cx=6.10, cy=-2.85, yaw_angle=114.7)
    build_speed_bump(cx=5.25, cy=-5.15, yaw_angle=0.0)

    add_cylinder(pos=[3.7, -5.5, 0.9], radius=0.25, half_height=1.9, rgba=self.cyl_silver)
    add_cylinder(pos=[1.9, -5.5, 0.9], radius=0.25, half_height=2.9, rgba=self.cyl_silver)
    add_cylinder(pos=[0.5, -6.5, 0.9], radius=0.25, half_height=1.9, rgba=self.cyl_silver)

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

if __name__ == "__main__":
   import mujoco.viewer
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   terrain = TerrainEntity(TerrainEntityCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG), device=device)
   print("Cargando escenario completo...")
   mujoco.viewer.launch(terrain.spec.compile())

