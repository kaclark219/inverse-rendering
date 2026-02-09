blender_script = r'''
import bpy
import math
from mathutils import Vector

records = RECORDS_PLACEHOLDER

def direction_to_quat(dir_world):
    """
    Rotate so Blender light's local -Z points along dir_world.
    """
    d = Vector(dir_world).normalized()
    forward = Vector((0.0, 0.0, -1.0))
    return forward.rotation_difference(d)

for rec in records:
    light_data = bpy.data.lights.new(name=rec["name"], type='SPOT')
    light_data.energy = rec["energy"]
    light_data.color = rec["color"]
    light_data.spot_size = math.radians(rec["spot_size_deg"])
    light_data.spot_blend = rec["spot_blend"]

    light_obj = bpy.data.objects.new(name=rec["name"], object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    light_obj.location = rec["location"]
    light_obj.rotation_mode = 'QUATERNION'
    light_obj.rotation_quaternion = direction_to_quat(rec["direction_world"])

print("Created", len(records), "lights.")
'''
print(blender_script.replace("RECORDS_PLACEHOLDER", repr(records)))