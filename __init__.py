bl_info = {
    "name": "Grid snap",
    "author": "Your Name Here",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "3D View",
    "description": "Example Add-on",
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "3D View",
}

import bpy, mathutils, bmesh, math

def create_transformed_empty(context, matrix: mathutils.Matrix):
    empty = bpy.data.objects.new(name = "Grid Origin Empty", object_data = None)

    empty.matrix_world = matrix.inverted()

    context.scene.collection.objects.link(empty)

    return empty

# The distinction between this and `apply_matrix_to_misc_view` is that properties modified in here are expected to be part of undo history, while those modified in `apply_matrix_to_misc_view` are not.
def apply_matrix_to_misc_scene(context, matrix):
    matrix = matrix.inverted()
    translation, rotation, _ = matrix.decompose()
    rotation_matrix = rotation.to_matrix()

    # Correct the scene's light direction (used for shadows)
    light_direction_matrix = mathutils.Matrix(
        (
            (-1, 0, 0),
            (0, 0, 1),
            (0, -1, 0),
        )
    )
    context.scene.display.light_direction = light_direction_matrix.inverted() @ rotation_matrix @ light_direction_matrix @ context.scene.display.light_direction


def apply_matrix_to_misc_view(context, matrix, interpolated = True):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences

    matrix = matrix.inverted()

    translation, rotation, _ = matrix.decompose()
    rotation_matrix = rotation.to_matrix()

    # Correct the cursor
    context.scene.cursor.matrix = matrix @ context.scene.cursor.matrix

    # Correct the active 3D View's view
    for area in [context.area]:
        space = area.spaces[0]

        if space.type != 'VIEW_3D':
            continue

        for region in area.regions:
            if region.type != 'WINDOW':
                continue

            order = "XZY"

            view_roll = region.data.view_rotation.to_euler(order).y

            region.data.view_location = rotation_matrix @ region.data.view_location
            region.data.view_location += translation

            if region.data.is_orthographic_side_view:
                continue

            region.data.view_rotation = rotation @ region.data.view_rotation

            if not addon_prefs.reset_roll:
                continue

            view_rotation_euler = region.data.view_rotation.to_euler(order)
            roll_amount = -view_rotation_euler.y + view_roll + 0.1

            print(f"Roll: {view_roll} -> {view_rotation_euler.y} -> offset {roll_amount} => {view_rotation_euler.y + roll_amount}")
            print(view_rotation_euler)

            if not interpolated or True:
                view_rotation_euler.y += roll_amount
                region.data.view_rotation = view_rotation_euler.to_quaternion()
                continue

            region.data.update()

            with context.temp_override(view = space, region = region):
                # bpy.ops.view3d.view_roll('INVOKE_REGION_WIN', angle = -roll_amount)
                bpy.ops.view3d.view_roll('EXEC_REGION_WIN', angle = -roll_amount)
            print(region.data.view_rotation.to_euler(order))

def clear_grid_transform(context):
    if context.scene.grid_origin is None:
        return

    parent = context.scene.grid_origin

    # Backwards, because the empty's transform is the inverse of the current grid transform!
    parent_matrix_inverted = parent.matrix_world
    parent_matrix = parent.matrix_world.inverted()

    context.scene.grid_origin = None

    context.scene.collection.objects.unlink(parent)

    for object in context.scene.objects:
        if object == parent:
            continue

        if object.parent == parent:
            object.parent = None
            continue

        object.matrix_world = parent_matrix @ object.matrix_world

    apply_matrix_to_misc_scene(context, parent_matrix_inverted)
    apply_matrix_to_misc_view(context, parent_matrix_inverted)

def set_grid_transform(context, transform: mathutils.Matrix):
    if context.scene.grid_origin is not None:
        clear_grid_transform(context)

    assert context.scene.grid_origin is None

    parent = create_transformed_empty(context, transform)
    context.scene.grid_origin = parent

    for object in context.scene.objects:
        if object == parent:
            continue
        
        object.parent = parent

    apply_matrix_to_misc_scene(context, transform)
    apply_matrix_to_misc_view(context, transform)

bpy.types.Scene.grid_origin = bpy.props.PointerProperty(type=bpy.types.Object, name="Grid Origin", description="The Empty currently set as the Grid Origin. Its transform is the inverse transform of the current grid transform", options=set())

class GridSnapAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    reset_roll: bpy.props.BoolProperty(
        name="Reset roll",
        description="Roll the camera so that it points up whenever the grid is changed. This makes navigation smoother if your orbit method is Turntable; this orientation change can be disorienting",
        default=True
   )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "reset_roll")

class SetGridOriginFromObject(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_object"
    bl_label = "Set grid origin to object"

    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object != context.scene.grid_origin

    def execute(self, context):
        clear_grid_transform(context)

        dg = context.evaluated_depsgraph_get()
        set_grid_transform(context, context.active_object.evaluated_get(dg).matrix_world)

        return {'FINISHED'}

class SetGridOriginFromFace(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_face"
    bl_label = "Set grid origin to center on face"
    bl_description = "Position the center of the grid on this face, with its front side facing +Z"

    bl_options = {'REGISTER', 'UNDO'}

    twist: bpy.props.IntProperty(name="Twist")

    @classmethod
    def poll(cls, context):
        if context.mode != 'EDIT_MESH': return False

        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        if bm.faces.active is None: return False

        bm.free()

        return True

    def execute(self, context):
        clear_grid_transform(context)

        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        face = bm.faces.active

        origin = face.calc_center_median()
        up = face.normal.normalized()

        front_edge = face.edges[self.twist % len(face.edges)]
        front = front_edge.verts[1].co - front_edge.verts[0].co

        back = -front

        right = back.cross(up).normalized()
        back = up.cross(right).normalized()

        mat = mathutils.Matrix((
            mathutils.Vector((*right, 0)),
            mathutils.Vector((*back, 0)),
            mathutils.Vector((*up, 0)),
            origin.to_4d(),
        )).transposed()

        dg = context.evaluated_depsgraph_get()
        mat = context.active_object.evaluated_get(dg).matrix_world @ mat

        set_grid_transform(context, mat)

        return {'FINISHED'}

class ClearGridOrigin(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_clear"
    bl_label = "Clear grid origin"

    @classmethod
    def poll(cls, context):
        return context.scene.grid_origin is not None

    def execute(self, context):
        clear_grid_transform(context)

        return {'FINISHED'}

def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(SetGridOriginFromObject.bl_idname)
    self.layout.operator(SetGridOriginFromFace.bl_idname)
    self.layout.operator(ClearGridOrigin.bl_idname)

history_pre_matrix = None
def history_pre_handler(scene):
    global history_pre_matrix

    if scene.grid_origin:
        history_pre_matrix = scene.grid_origin.matrix_world
    else:
        history_pre_matrix = mathutils.Matrix()

    print("Before re/undo:", scene, scene.grid_origin)

def history_post_handler(scene):
    global history_pre_matrix
    
    if scene.grid_origin:
        matrix = scene.grid_origin.matrix_world.inverted()
    else:
        matrix = mathutils.Matrix()

    apply_matrix_to_misc_view(bpy.context, history_pre_matrix @ matrix, interpolated = False)

    print("After re/undo:", scene, scene.grid_origin)

def register():
    bpy.utils.register_class(GridSnapAddonPreferences)
    bpy.utils.register_class(SetGridOriginFromObject)
    bpy.utils.register_class(SetGridOriginFromFace)
    # bpy.utils.register_class(SetGridOriginFront)
    bpy.utils.register_class(ClearGridOrigin)
    bpy.types.VIEW3D_MT_view.append(menu_func)

    bpy.app.handlers.undo_pre.append(history_pre_handler)
    bpy.app.handlers.undo_post.append(history_post_handler)

    bpy.app.handlers.redo_pre.append(history_pre_handler)
    bpy.app.handlers.redo_post.append(history_post_handler)

def unregister():
    bpy.utils.unregister_class(GridSnapAddonPreferences)
    bpy.utils.unregister_class(SetGridOriginFromObject)
    bpy.utils.unregister_class(SetGridOriginFromFace)
    # bpy.utils.unregister_class(SetGridOriginFront)
    bpy.utils.unregister_class(ClearGridOrigin)
    bpy.types.VIEW3D_MT_view.remove(menu_func)

    bpy.app.handlers.undo_pre.remove(history_pre_handler)
    bpy.app.handlers.undo_post.remove(history_post_handler)

    bpy.app.handlers.redo_pre.remove(history_pre_handler)
    bpy.app.handlers.redo_post.remove(history_post_handler)
