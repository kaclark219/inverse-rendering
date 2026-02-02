import bpy, csv, os
from mathutils import Vector
import math

# SETUP
OUT_CSV = bpy.path.abspath("//master_with_paths.csv") # relative to the blend file
# folders for dataset
DATASET_ROOT = "../data"
MATERIAL_FOLDER = "PlasticGlossy"
SHAPES = ["Cone","Cube","Cylinder","Icosphere","Sphere","Torus"]
# # of camera renders per lighting/material configuration
CAMERAS_PER_CONFIG = 17
# max # of brightest lights to record per frame
MAX_ACTIVE_LIGHTS = 3

# map object names to light types
LIGHTS_SINGLE = {"Area Light":"Area", "Point Light":"Point", "Spot Light":"Spot"}
LIGHTS_TRI = ["TriLamp-Key","TriLamp-Fill","TriLamp-Back"]
ALL_LIGHT_NAMES = list(LIGHTS_SINGLE.values()) + LIGHTS_TRI

# HELPER FUNCTIONS
def safe_float(x):
    """Safely cast to float, returning empty string if conversion fails."""
    try: return float(x)
    except: return ""

def safe_str(x):
    """Safely cast to string, returning empty string if conversion fails."""
    try: return str(x)
    except: return ""

def obj_forward_world(obj):
    """Compute the object's forward (-Z) direction in world space."""
    return (obj.matrix_world.to_3x3() @ Vector((0,0,-1))).normalized()

def set_light_setup(setup_name):
    """Enable the desired light rig and disable all others.

    Args:
        setup_name: One of the names in `setups` (e.g., "Point Light", "Tri Light").
    """
    # turn all known lights off first
    for name in ALL_LIGHT_NAMES:
        obj = bpy.data.objects.get(name)
        if obj: obj.hide_render = True
    # enable the single named lamp
    if setup_name in LIGHTS_SINGLE:
        obj = bpy.data.objects.get(LIGHTS_SINGLE[setup_name])
        if obj: obj.hide_render = False
    # enable the 3-point lighting setup
    elif setup_name == "Tri Light":
        for name in LIGHTS_TRI:
            obj = bpy.data.objects.get(name)
            if obj: obj.hide_render = False
    # all lamps off .. world lighting only
    elif setup_name.startswith("HDRI"):
        pass

def get_engine_and_batch(scene):
    """Collect render engine settings and derive a batch folder name."""
    eng = safe_str(scene.render.engine).lower()
    eng_name = "Cycles" if "cycles" in eng else ("Eevee" if "eevee" in eng else "UnknownEngine")

    vt = safe_str(scene.view_settings.view_transform).lower()
    vt_name = "AGX" if "agx" in vt else ("Filmic" if "filmic" in vt else "UnknownView")

    return {
        "render_engine": safe_str(scene.render.engine),
        "view_transform": safe_str(scene.view_settings.view_transform),
        "look": safe_str(scene.view_settings.look),
        # folder name used in the dataset layout
        "batch_folder": f"Batch 1 - {eng_name} {vt_name}",
    }

def get_camera(scene):
    """Extract camera position, orientation vectors, and lens data."""
    cam = scene.camera
    if not cam: return {}
    cam_pos = cam.matrix_world.translation
    cam_rot = cam.matrix_world.to_3x3()
    cam_forward = (cam_rot @ Vector((0,0,-1))).normalized()
    cam_up = (cam_rot @ Vector((0,1,0))).normalized()
    cam_right = (cam_rot @ Vector((1,0,0))).normalized()
    return {
        "camera_name": safe_str(cam.name),
        "cam_pos_x": safe_float(cam_pos.x), "cam_pos_y": safe_float(cam_pos.y), "cam_pos_z": safe_float(cam_pos.z),
        "cam_forward_x": safe_float(cam_forward.x), "cam_forward_y": safe_float(cam_forward.y), "cam_forward_z": safe_float(cam_forward.z),
        "cam_up_x": safe_float(cam_up.x), "cam_up_y": safe_float(cam_up.y), "cam_up_z": safe_float(cam_up.z),
        "cam_right_x": safe_float(cam_right.x), "cam_right_y": safe_float(cam_right.y), "cam_right_z": safe_float(cam_right.z),
        "focal_length_mm": safe_float(cam.data.lens) if cam.data else "",
    }

def get_active_lights(scene, max_n=MAX_ACTIVE_LIGHTS):
    """Return info for the brightest visible lights in the scene.

    The list is sorted by energy and capped to `max_n`.
    """
    cam = scene.camera
    cam_rot = cam.matrix_world.to_3x3() if cam else None
    lights = []
    # collect active lights (non-zero energy)
    for obj in bpy.data.objects:
        if obj.type != "LIGHT": continue
        if obj.hide_render: continue
        energy = float(getattr(obj.data, "energy", 0.0))
        if energy <= 0.0: continue
        lights.append(obj)
    # sort by energy descending
    lights.sort(key=lambda o: float(getattr(o.data, "energy", 0.0)), reverse=True)
    lights = lights[:max_n]

    out = {"num_active_lights": len(lights)}
    for i in range(max_n):
        p = f"light{i}_"
        # if no light for this slot, emit empty values
        if i >= len(lights):
            out.update({p+k:"" for k in ["name","type","energy","color_r","color_g","color_b",
                                        "pos_x","pos_y","pos_z","dir_x","dir_y","dir_z",
                                        "dir_cam_x","dir_cam_y","dir_cam_z",
                                        "spot_cone_deg","spot_blend",
                                        "area_shape","area_size_x","area_size_y"]})
            continue

        obj = lights[i]
        L = obj.data
        # world-space light position & direction
        pos = obj.matrix_world.translation
        dir_world = obj_forward_world(obj)
        # light direction in camera space
        dir_cam = (cam_rot.inverted() @ dir_world).normalized() if cam_rot else Vector((0,0,0))
        col = getattr(L, "color", (1,1,1))

        out[p+"name"] = obj.name
        out[p+"type"] = L.type
        out[p+"energy"] = safe_float(getattr(L,"energy",0.0))
        out[p+"color_r"] = safe_float(col[0]); out[p+"color_g"] = safe_float(col[1]); out[p+"color_b"] = safe_float(col[2])
        out[p+"pos_x"] = safe_float(pos.x); out[p+"pos_y"] = safe_float(pos.y); out[p+"pos_z"] = safe_float(pos.z)
        out[p+"dir_x"] = safe_float(dir_world.x); out[p+"dir_y"] = safe_float(dir_world.y); out[p+"dir_z"] = safe_float(dir_world.z)
        out[p+"dir_cam_x"] = safe_float(dir_cam.x); out[p+"dir_cam_y"] = safe_float(dir_cam.y); out[p+"dir_cam_z"] = safe_float(dir_cam.z)

        # spotlight properties
        if L.type == "SPOT":
            out[p+"spot_cone_deg"] = safe_float(math.degrees(L.spot_size))
            out[p+"spot_blend"] = safe_float(L.spot_blend)
        else:
            out[p+"spot_cone_deg"] = ""; out[p+"spot_blend"] = ""

        # area light properties
        if L.type == "AREA":
            out[p+"area_shape"] = safe_str(getattr(L,"shape",""))
            out[p+"area_size_x"] = safe_float(getattr(L,"size",0.0))
            if out[p+"area_shape"] in {"RECTANGLE","ELLIPSE"}:
                out[p+"area_size_y"] = safe_float(getattr(L,"size_y",0.0))
            else:
                out[p+"area_size_y"] = out[p+"area_size_x"]
        else:
            out[p+"area_shape"] = ""; out[p+"area_size_x"] = ""; out[p+"area_size_y"] = ""

    return out

# MAIN EXECUTION LOGIC
# grab current scene & frame range
scene = bpy.context.scene
fs, fe = scene.frame_start, scene.frame_end

# headers for the csv file
header = ["image_relpath","image_exists","shape_name","material_folder","light_folder","batch_folder",
    "frame","config_id","camera_png",
    "render_engine","view_transform","look",
    "camera_name",
    "cam_pos_x","cam_pos_y","cam_pos_z",
    "cam_forward_x","cam_forward_y","cam_forward_z",
    "cam_up_x","cam_up_y","cam_up_z",
    "cam_right_x","cam_right_y","cam_right_z",
    "focal_length_mm",
    "num_active_lights"]
# columns per light
for i in range(MAX_ACTIVE_LIGHTS):
    p = f"light{i}_"
    header += [p+"name",p+"type",p+"energy",
                p+"color_r",p+"color_g",p+"color_b",
                p+"pos_x",p+"pos_y",p+"pos_z",
                p+"dir_x",p+"dir_y",p+"dir_z",
                p+"dir_cam_x",p+"dir_cam_y",p+"dir_cam_z",
                p+"spot_cone_deg",p+"spot_blend",
                p+"area_shape",p+"area_size_x",p+"area_size_y"]

# different lighting setups
setups = ["Point Light","Spot Light","Area Light","Tri Light","HDRI (Sunlight)","HDRI (Overcast)"]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()

    # iterate through each lighting setup
    for setup in setups:
        set_light_setup(setup)

        # iterate through each frame in the scene
        for frame in range(fs, fe+1):
            scene.frame_set(frame)

            # derive camera index & configuration id from frame index
            idx = frame - fs
            camera_png = (idx % CAMERAS_PER_CONFIG) + 1
            config_id  = idx // CAMERAS_PER_CONFIG

            # base row data for all shapes in this frame/setup
            base = {"material_folder": MATERIAL_FOLDER,
                    "light_folder": setup,
                    "frame": frame, "config_id": config_id, "camera_png": camera_png}
            base.update(get_engine_and_batch(scene))
            base.update(get_camera(scene))
            base.update(get_active_lights(scene))

            # emit a row per shape
            for shape in SHAPES:
                rel = os.path.join(shape, MATERIAL_FOLDER, setup, base["batch_folder"], f"{camera_png}.png")
                exists = ""
                if DATASET_ROOT.strip():
                    exists = os.path.exists(os.path.join(DATASET_ROOT, rel))
                row = dict(base)
                row["shape_name"] = shape
                row["image_relpath"] = rel
                row["image_exists"] = exists
                w.writerow(row)

print("Wrote:", OUT_CSV)