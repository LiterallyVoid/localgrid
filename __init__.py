bl_info = {
    "name": "Grid Snap",
    "author": "LiterallyVoid",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "3D View",
    "description": "",
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "3D View",
}

import bpy, mathutils, bmesh, math
from bpy.app.handlers import persistent

grid_origin_empty_name = "Grid Snap: Grid Origin Empty"

def create_transformed_empty(context, matrix: mathutils.Matrix):
    try:
        empty = bpy.data.objects[grid_origin_empty_name]
        if empty.users > 0:
            raise ValueError
    except:
        empty = bpy.data.objects.new(name = grid_origin_empty_name, object_data = None)

    empty.matrix_world = matrix.inverted()

    if empty.parent:
        # This happened in testing. I hope that it was because of a transient state being saved because of an addon crash, but if it ever happens then it will be very bad. (The exact issue was that this empty was being parented to itself, which indicates that a lot of things had gone very wrong in Blender's data model.)
        print("This should never happen!")
        empty.parent = None

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
    for screen in context.workspace.screens:
        for area in screen.areas:
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

                old = region.data.view_rotation
                naive = rotation @ region.data.view_rotation

                if not addon_prefs.reset_roll:
                    region.data.view_rotation = naive
                    continue

                forwards = region.data.view_rotation @ mathutils.Vector((0, 0, 1))

                forwards = rotation @ forwards

                tracked = forwards.to_track_quat('Z', 'Y')

                previous_roll = old.to_euler(order).y
                # tracked = tracked @ mathutils.Quaternion((0, 1, 0), -previous_roll)

                delta_from_naive = naive.inverted() @ tracked
                roll = -delta_from_naive.to_euler(order).z

                region.data.view_rotation = naive

                region.data.update()

                # FIXME: Interpolation doesn't work well at all.
                if interpolated and False:
                    with context.temp_override(screen = screen, area = area, view = space, region = region):
                        bpy.ops.view3d.view_roll('INVOKE_REGION_WIN', angle = -roll)
                else:
                    region.data.view_rotation = tracked

def clear_grid_transform(context):
    if context.scene.grid_origin is None:
        return

    parent = context.scene.grid_origin

    # Backwards, because the empty's transform is the inverse of the current grid transform!
    grid_matrix_inverted = parent.matrix_world
    grid_matrix = parent.matrix_world.inverted()

    context.scene.grid_origin = None

    for object in context.scene.objects:
        if object == parent:
            continue

        if object.parent == parent:
            object.parent = None
            continue

        if object.parent is not None:
            continue

        object.matrix_world = grid_matrix @ object.matrix_world

    context.scene.collection.objects.unlink(parent)

    apply_matrix_to_misc_scene(context, grid_matrix_inverted)
    apply_matrix_to_misc_view(context, grid_matrix_inverted)

# Remove the scale component of `matrix`
def remove_scale(matrix: mathutils.Matrix) -> mathutils.Matrix:
    translation, rotation, _ = matrix.decompose()
    return mathutils.Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()

# Reduce the rotation of `matrix`, so that it has minimal deflection of the Z axis.
def reduce_transform(matrix: mathutils.Matrix) -> mathutils.Matrix:
    translation, rotation, _ = matrix.decompose()

    cardinal_axes = [
        mathutils.Quaternion(),
        mathutils.Quaternion((1, 0, 0), math.pi * 0.5),
        mathutils.Quaternion((1, 0, 0), math.pi),
        mathutils.Quaternion((1, 0, 0), math.pi * 1.5),
        mathutils.Quaternion((0, 1, 0), math.pi * 0.5),
        mathutils.Quaternion((0, 1, 0), math.pi * 1.5),
    ]

    scored = []

    for mod in cardinal_axes:
        up = rotation @ mod @ mathutils.Vector((0, 0, 1))
        score = up.dot((0, 0, 1))

        scored.append((matrix @ mod.to_matrix().to_4x4() , score))

    scored.sort(key = lambda tup: tup[1])
    
    return scored[-1][0]

def set_grid_transform(context, transform: mathutils.Matrix):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences

    transform = remove_scale(transform)

    if addon_prefs.minimize_roll:
        transform = reduce_transform(transform)

    if context.scene.grid_origin is not None:
        clear_grid_transform(context)

    assert context.scene.grid_origin is None

    parent = create_transformed_empty(context, transform)
    context.scene.grid_origin = parent

    for object in context.scene.objects:
        if object == parent:
            continue

        if object.parent is not None:
            continue
        
        object.parent = parent

    apply_matrix_to_misc_scene(context, transform)
    apply_matrix_to_misc_view(context, transform)

    if addon_prefs.move_cursor_to_origin:
        context.scene.cursor.matrix = mathutils.Matrix()

bpy.types.Scene.grid_origin = bpy.props.PointerProperty(type=bpy.types.Object, name="Grid Origin", description="The Empty currently set as the Grid Origin. Its transform is the inverse transform of the current grid transform", options=set())

class GridSnapAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    reset_roll: bpy.props.BoolProperty(
        name="Reset Roll",
        description="Roll the camera so that it points up whenever the grid is changed. This makes navigation smoother if your orbit method is Turntable; this orientation change can be disorienting",
        default=True
    )

    minimize_roll: bpy.props.BoolProperty(
        name="Minimize Roll",
        description="Instead of always aligning the +Z axis to the object or face, align whichever axis requires the least roll",
        default=True
    )

    move_cursor_to_origin: bpy.props.BoolProperty(
        name="Move Cursor to Origin",
        description="Move the cursor to the origin whenever the grid origin is set",
        default=True
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "reset_roll")
        layout.prop(self, "minimize_roll")
        layout.prop(self, "move_cursor_to_origin")

def matrix_from_axes(
    origin: mathutils.Vector,
    up_vector: mathutils.Vector,
    front_vector: mathutils.Vector,
) -> mathutils.Matrix:
    back = -front_vector

    right = back.cross(up_vector).normalized()
    back = up_vector.cross(right).normalized()

    mat = mathutils.Matrix((
        mathutils.Vector((*right, 0)),
        mathutils.Vector((*back, 0)),
        mathutils.Vector((*up_vector, 0)),
        origin.to_4d(),
    )).transposed()

    return mat

def bmesh_face_axes(
    face: bmesh.types.BMFace,
    edge_index: int,
    align_vertex_instead_of_edge: bool,
) -> (mathutils.Vector, mathutils.Vector):
    origin = face.calc_center_median()
    up = face.normal.normalized()

    longest_edge = 0
    longest_edge_length = 0

    for i, edge in enumerate(face.edges):
        edge_length = (edge.verts[1].co - edge.verts[0].co).magnitude
        if edge_length <= longest_edge_length:
            continue

        longest_edge = i
        longest_edge_length = edge_length

    front_edge = face.edges[edge_index % len(face.edges)]
    front = front_edge.verts[1].co - front_edge.verts[0].co

    if align_vertex_instead_of_edge:
        front = front_edge.verts[0].co - origin

    return (origin, up, front)

GLOBAL_visualization_gizmo_handle = None

class SetGridOrigin(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set"
    bl_label = "Set Grid Origin"
    bl_description = "Set grid origin from active object, face, vertex, or bone"

    bl_options = {'REGISTER', 'UNDO'}

    align_to: bpy.props.EnumProperty(
        name="Align to",
        items=[
            ('DEFAULT', "#DEFAULT!", "..."),
            ('OBJECT', "Object", "Align grid to the active object"),
            ('FACE_VERTEX', "Face Vertex", "Align grid to the active face with one vertex lying on a cardinal axis, starting from the first vertex of the longest edge"),
            ('FACE_EDGE', "Face Edge", "Align grid to the active face with one of its edges along a cardinal axis, starting from the longest edge"),
            ('BONE', "Bone", "Align grid to the active bone")
        ],
        default='DEFAULT',
        options={'SKIP_SAVE'},
    )

    face_twist: bpy.props.IntProperty(name="Twist", options={'SKIP_SAVE'})
    face_angle_offset: bpy.props.FloatProperty(name="Angle Offset", unit='ROTATION', options={'SKIP_SAVE'})

    bone_head_tail: bpy.props.FloatProperty(name="Head/Tail", min=0.0, max=1.0, subtype="FACTOR")

    _draw_handle: object = None

    @staticmethod
    def poll_object(context):
        return context.active_object is not None

    @staticmethod
    def poll_face(context):
        # My intuition is that going from edit mesh -> bmesh is expensive, so skip it and just assume there'll be an active face. This may be wrong!
        return context.mode == 'EDIT_MESH' 

    @staticmethod
    def poll_bone(context):
        # Assumption: context.active_pose_bone is `True` iff there's an active bone and the context is in pose mode.
        return (context.mode == 'EDIT_ARMATURE' or context.mode == 'POSE') and context.active_bone is not None

    @classmethod
    def poll(cls, context):
        return cls.poll_object(context) or cls.poll_face(context) or cls.poll_bone(context)

    def draw(self, context):
        row = self.layout.row(align = True)

        if self.poll_object(context):
            row.prop_enum(self, "align_to", 'OBJECT')


        if self.poll_face(context):
            row.prop_enum(self, "align_to", 'FACE_VERTEX')
            row.prop_enum(self, "align_to", 'FACE_EDGE')

            layout = self.layout.column()

            layout.prop(self, "face_twist")
            layout.prop(self, "face_angle_offset")

            layout.enabled = self.align_to == 'FACE_VERTEX' or self.align_to == 'FACE_EDGE'


        if self.poll_bone(context):
            row.prop_enum(self, "align_to", 'BONE')

            layout = self.layout.column()

            layout.prop(self, "bone_head_tail")

            layout.enabled = self.align_to == 'BONE'

    def execute(self, context):
        if self.align_to == 'DEFAULT':
            match context.mode:
                case 'OBJECT':
                    self.align_to = 'OBJECT'
                case 'EDIT_MESH':
                    self.align_to = 'FACE_EDGE'
                case 'EDIT_ARMATURE' | 'POSE':
                    self.align_to = 'BONE'
        clear_grid_transform(context)

        dg = context.evaluated_depsgraph_get()

        # The active object's matrix is required for every snap mode.
        active_object_matrix = context.active_object.evaluated_get(dg).matrix_world

        matrix = None

        match self.align_to:
            case 'DEFAULT': assert False
            case 'OBJECT':
                matrix = active_object_matrix
            case 'FACE_VERTEX' | 'FACE_EDGE':
                data = context.active_object.data
                bm = bmesh.from_edit_mesh(data)
                face = bm.faces.active

                if face is None:
                    matrix = active_object_matrix
                    self.align_to = 'OBJECT'
                else:
                    origin, up, front = bmesh_face_axes(
                        face,
                        self.face_twist,

                        # align_vertex_instead_of_edge:
                        self.align_to == 'FACE_VERTEX', 
                    )
                    matrix = active_object_matrix @ matrix_from_axes(origin, up, front)
                    matrix = matrix @ mathutils.Matrix.Rotation(self.face_angle_offset, 4, 'Z')

                bm.free()
            case 'BONE':
                bone_matrix = None
                bone_length = 0

                if context.active_pose_bone is not None:
                    bone_matrix = context.active_pose_bone.matrix
                    bone_length = context.active_pose_bone.length
                elif context.active_bone is not None and isinstance(context.active_bone, bpy.types.EditBone):
                    bone_matrix = context.active_bone.matrix
                    bone_length = context.active_bone.length
                else:
                    bone_matrix = context.active_bone.matrix_local
                    bone_length = context.active_bone.length

                bone_matrix = bone_matrix @ mathutils.Matrix.Translation((0, bone_length * self.bone_head_tail, 0))

                matrix = active_object_matrix @ bone_matrix

        assert matrix is not None

        set_grid_transform(context, matrix)

        return {'FINISHED'}

class ClearGridOrigin(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_clear"
    bl_label = "Reset Grid Origin"
    bl_description = "Reset grid back to the scene's origin"

    @classmethod
    def poll(cls, context):
        return context.scene.grid_origin is not None

    def execute(self, context):
        clear_grid_transform(context)

        return {'FINISHED'}

history_pre_matrix = None

@persistent
def history_pre_handler(scene):
    global history_pre_matrix

    if scene.grid_origin:
        history_pre_matrix = scene.grid_origin.matrix_world
    else:
        history_pre_matrix = mathutils.Matrix()

@persistent
def history_post_handler(scene):
    global history_pre_matrix
    
    if scene.grid_origin:
        matrix = scene.grid_origin.matrix_world.inverted()
    else:
        matrix = mathutils.Matrix()

    apply_matrix_to_misc_view(bpy.context, history_pre_matrix @ matrix, interpolated = True)

def menu_func(self, context):
    self.layout.separator()
    self.layout.operator(SetGridOrigin.bl_idname)
    self.layout.operator(ClearGridOrigin.bl_idname)

def register():
    bpy.utils.register_class(GridSnapAddonPreferences)
    bpy.utils.register_class(SetGridOrigin)
    bpy.utils.register_class(ClearGridOrigin)
    bpy.types.VIEW3D_MT_view.append(menu_func)

    bpy.app.handlers.undo_pre.append(history_pre_handler)
    bpy.app.handlers.undo_post.append(history_post_handler)

    bpy.app.handlers.redo_pre.append(history_pre_handler)
    bpy.app.handlers.redo_post.append(history_post_handler)

def unregister():
    bpy.utils.unregister_class(GridSnapAddonPreferences)
    bpy.utils.unregister_class(SetGridOrigin)
    bpy.utils.unregister_class(ClearGridOrigin)
    bpy.types.VIEW3D_MT_view.remove(menu_func)

    bpy.app.handlers.undo_pre.remove(history_pre_handler)
    bpy.app.handlers.undo_post.remove(history_post_handler)

    bpy.app.handlers.redo_pre.remove(history_pre_handler)
    bpy.app.handlers.redo_post.remove(history_post_handler)
