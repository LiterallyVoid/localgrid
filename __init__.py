bl_info = {
    "name": "Local Grid",
    "author": "LiterallyVoid",
    "version": (1, 0),
    "blender": (4, 0, 0),
    "location": "View3D > View",
    "description": "",
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "3D View",
}

import bpy, mathutils, bmesh, math
from typing import Optional
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
        # This maybe happened in testing. I hope that it was either user error or because of a transient state being saved because of an addon crash, but if it ever happens then it will be very bad. (The exact issue was that this empty was being parented to itself, which indicates that a lot of things had gone very wrong in Blender's data model.)
        print("This should never happen!")
        empty.parent = None

    context.scene.collection.objects.link(empty)

    empty.select_set(False)
    empty.hide_set(True)

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
            try:
                space = area.spaces[0]
            except:
                # ???
                continue

            if space.type != 'VIEW_3D':
                continue

            for region in area.regions:
                if region.type != 'WINDOW':
                    continue

                # @TODO: find out why this happens
                if not hasattr(region.data, "view_rotation"):
                    print(f"Region has no `view_rotation` property? {region}, {region.data}, {space}, {area}")
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
                if interpolated:
                    with context.temp_override(screen = screen, area = area, view = space, region = region):
                        bpy.ops.view3d.view_roll('INVOKE_REGION_WIN', angle = -roll)
                else:
                    region.data.view_rotation = tracked

# Clear grid transform, and return what it was as a Matrix.
def clear_grid_transform(context, interpolated = True):
    if context.scene.grid_origin is None:
        return mathutils.Matrix()

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
    apply_matrix_to_misc_view(context, grid_matrix_inverted, interpolated)

    return grid_matrix

# Remove the scale component of `matrix`
def remove_scale(matrix: mathutils.Matrix) -> mathutils.Matrix:
    translation, rotation, _ = matrix.decompose()
    return mathutils.Matrix.Translation(translation) @ rotation.to_matrix().to_4x4()

# Reduce the rotation of `matrix`, so that it has minimal deflection of the Z axis.
def reduce_transform(matrix: mathutils.Matrix, previous_matrix: mathutils.Matrix) -> mathutils.Matrix:
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

    # @TODO: max(..., key = ...)
    for i, mod in enumerate(cardinal_axes):
        up = rotation @ mod @ mathutils.Vector((0, 0, 1))

        score = up.dot((0, 0, 1))

        # Prefer no orientation change.
        if i == 0: score += 0.1

        scored.append((matrix @ mod.to_matrix().to_4x4(), score))

    scored.sort(key = lambda tup: tup[1])
    
    return scored[-1][0]

# `previous_matrix` will be used to minimize roll.
def set_grid_transform(context, transform: mathutils.Matrix, previous_matrix: Optional[mathutils.Matrix] = None, interpolated = True):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences

    transform = remove_scale(transform)

    new_up = transform.col[2].to_3d()
    new_side = transform.col[0].to_3d()

    if addon_prefs.minimize_roll:
        transform = reduce_transform(transform, previous_matrix)

    if context.scene.grid_origin is not None:
        clear_grid_transform(context, interpolated)

    assert context.scene.grid_origin is None

    parent = create_transformed_empty(context, transform)
    context.scene.grid_origin = parent
    context.scene.grid_origin_up = new_up
    context.scene.grid_origin_side = new_side

    for object in context.scene.objects:
        if object == parent:
            continue

        if object.parent is not None:
            continue
        
        object.parent = parent

    apply_matrix_to_misc_scene(context, transform)
    apply_matrix_to_misc_view(context, transform, interpolated)

    if addon_prefs.move_cursor_to_origin:
        context.scene.cursor.matrix = mathutils.Matrix()

bpy.types.Scene.grid_origin = bpy.props.PointerProperty(type=bpy.types.Object, name="Grid Origin", description="The Empty currently set as the Grid Origin. Its transform is the inverse transform of the current grid transform", options=set())
bpy.types.Scene.grid_origin_up = bpy.props.FloatVectorProperty(name="Grid +Z Axis, in world space", description="May differ from the current grid transform if Minimize Roll is set", subtype='DIRECTION')
bpy.types.Scene.grid_origin_side = bpy.props.FloatVectorProperty(name="Some vector perpendicular to `grid_origin_up`, in world space", description="May differ from the current grid transform if Minimize Roll is set", subtype='DIRECTION')

class GridSnapAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    reset_roll: bpy.props.BoolProperty(
        name="Reset Roll",
        description="Roll the camera so that it points up whenever the grid is changed. This makes navigation easier if your orbit method is Turntable; otherwise it can be disorienting",
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

def choose_orthogonal_axis(orthogonal_to):
    orthogonal_to = orthogonal_to.normalized()

    axes = [
        mathutils.Vector((1, 0, 0)),
        mathutils.Vector((0, 1, 0)),
        mathutils.Vector((0, 0, 1)),
    ]

    return min(axes, key = lambda axis: abs(axis.dot(orthogonal_to)))


def matrix_from_axes(
    origin: mathutils.Vector,
    up_vector: mathutils.Vector,
    front_vector: Optional[mathutils.Vector],
) -> mathutils.Matrix:
    up_vector = up_vector.normalized()

    if front_vector is None:
        # If there's no front vector given, just choose a random axis?
        axes = [
            mathutils.Vector((1, 0, 0)),
            mathutils.Vector((0, 1, 0)),
            mathutils.Vector((0, 0, 1)),
        ]

        # Choose the cardinal axis that's most cardinal from `up`.
        front_vector = min(axes, key = lambda axis: 1 - abs(axis.dot(up_vector)))

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

# If no `front_vector` is given, choose the longest edge.
def bmesh_face_axes(
    face: bmesh.types.BMFace,
    front_vector: Optional[mathutils.Vector],
) -> (mathutils.Vector, mathutils.Vector, mathutils.Vector):
    origin = face.calc_center_median()
    up = face.normal.normalized()

    if front_vector is None:
        longest_edge = 0
        longest_edge_length = 0

        for i, edge in enumerate(face.edges):
            edge_length = (edge.verts[1].co - edge.verts[0].co).magnitude
            if edge_length <= longest_edge_length:
                continue

            longest_edge = i
            longest_edge_length = edge_length

        edge_v0 = face.verts[longest_edge].co
        edge_v1 = face.verts[(longest_edge + 1) % len(face.verts)].co
        front_vector = edge_v1 - edge_v0

    return (origin, up, front_vector)

class SetGridOriginFromCursor(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_cursor"
    bl_label = "Set Local Grid from Cursor"
    bl_description = "Center grid around the 3D Cursor"

    bl_options = {'REGISTER', 'UNDO'}

    rotation: bpy.props.BoolProperty(name="Rotation", default=False)

    def execute(self, context):
        dg = context.evaluated_depsgraph_get()

        initial_matrix = clear_grid_transform(context, False)

        matrix = context.scene.cursor.matrix.copy()

        if not self.rotation:
            translation = matrix.to_translation()
            matrix = mathutils.Matrix.Translation(translation) @ initial_matrix.to_quaternion().to_matrix().to_4x4()

        set_grid_transform(context, matrix, initial_matrix)

        return {'FINISHED'}



class SetGridOriginFromCursor(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_cursor_project"
    bl_label = "Project Local Grid from Cursor"
    bl_description = "Rotate grid until the 3D Cursor lies on a cardinal axis"

    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dg = context.evaluated_depsgraph_get()

        initial_matrix = clear_grid_transform(context, False)

        if context.scene.grid_origin is None:
            up = mathutils.Vector((0, 0, 1))
        else:
            up = context.scene.grid_origin_up

        matrix = matrix_from_axes(center, up, axis)

        set_grid_transform(context, matrix, initial_matrix)

        return {'FINISHED'}


class SetGridOriginFromActive(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_active"
    bl_label = "Set Local Grid from Active"
    bl_description = "Center the grid around the active Object, Face, Edge, or Vertex"

    bl_options = {'REGISTER', 'UNDO'}

    rotation: bpy.props.BoolProperty(name="Rotation", default=True)

    @classmethod
    def poll(cls, context):
        if context.active_object is None: return False

        if context.mode == 'OBJECT': return True

        if cls.poll_mesh(context): return True
        if cls.poll_bone(context): return True

        return False


    @staticmethod
    def poll_mesh(context):
        if context.mode != 'EDIT_MESH': return False

        data = context.active_object.data

        bm = bmesh.from_edit_mesh(data)

        if bm.select_history.active is None:
            bm.free()
            return False

        bm.free()

        return True
        

    @staticmethod
    def poll_bone(context):
        if context.mode != 'EDIT_ARMATURE' and context.mode != 'POSE': return False
        return context.active_bone is not None


    def execute(self, context):
        dg = context.evaluated_depsgraph_get()

        initial_matrix = mathutils.Matrix()
        if context.scene.grid_origin is not None:
            initial_matrix = context.scene.evaluated_get(dg).grid_origin.matrix_world.inverted()

        initial_matrix = clear_grid_transform(context, interpolated = True)
        initial_rotation = initial_matrix.to_quaternion().to_matrix().to_4x4()

        active_object_matrix = context.active_object.evaluated_get(dg).matrix_world
        active_object_rotation = active_object_matrix.to_quaternion().to_matrix().to_4x4()
        matrix = None

        if self.poll_mesh(context):
            data = context.active_object.data
            bm = bmesh.from_edit_mesh(data)

            active = bm.select_history.active

            if isinstance(active, bmesh.types.BMFace):
                origin, up, front = bmesh_face_axes(
                    active,
                    # Just guess a front vector (from the longest edge?)
                    None,
                )

                matrix = active_object_matrix @ matrix_from_axes(origin, up, front)


            elif isinstance(active, bmesh.types.BMEdge):
                origin = (active.verts[0].co + active.verts[1].co) / 2
                origin = active_object_matrix @ origin

                up = active.verts[1].co - active.verts[0].co
                up = up.normalized()

                up = active_object_rotation @ up

                front = initial_rotation @ choose_orthogonal_axis(
                    initial_rotation @ up
                )

                print(up.dot(front))

                matrix = matrix_from_axes(origin, up, front)


            elif isinstance(active, bmesh.types.BMVert):
                translation = active_object_matrix @ active.co
                matrix = mathutils.Matrix.Translation(translation) @ initial_rotation


        elif self.poll_bone(context):
            bone_matrix = None
            bone_length = 0

            if context.active_pose_bone is not None:
                bone_matrix = context.active_pose_bone.matrix
                bone_length = context.active_pose_bone.length
            elif context.active_bone is not None and isinstance(context.active_bone, bpy.types.EditBone):
                bone_matrix = context.active_bone.matrix
                bone_length = context.active_bone.length
            else:
                # It doesn't make sense to reach here!

                # bone_matrix = context.active_bone.matrix_local
                # bone_length = context.active_bone.length

                raise ValueError("this should not happen") # If this happens, maybe uncommenting the above lines will help.

            # bone_matrix = bone_matrix @ mathutils.Matrix.Translation((0, bone_length * self.bone_head_tail, 0))

            # bone_matrix = mathutils.Matrix.Translation((0, 0, 0))

            matrix = active_object_matrix @ bone_matrix


        else:
            matrix = active_object_matrix

        set_grid_transform(context, matrix, initial_matrix, interpolated = True)

        return {'FINISHED'}

class SetGridOriginFromVertices(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_vertices"
    bl_label = "Set Grid Origin From Vertices"
    bl_description = "Set grid origin from three selected vertices"

    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.mode != 'EDIT_MESH': return False
        if context.active_object is None: return False

        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        if len(bm.select_history) != 3:
            return False

        if not all([isinstance(elem, bmesh.types.BMVert) for elem in bm.select_history]):
            return False

        return True

    def execute(self, context):
        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        center = bm.select_history.active.co
        front = bm.select_history[1].co - center
        adjacent = bm.select_history[0].co - center
        up = adjacent.cross(front)

        if up.magnitude < 1e-12:
            raise ValueError("colinear")

        dg = context.evaluated_depsgraph_get()

        initial_matrix = mathutils.Matrix()
        if context.scene.grid_origin is not None:
            initial_matrix = context.scene.evaluated_get(dg).grid_origin.matrix_world.inverted()

        clear_grid_transform(context, interpolated = True)

        active_object_matrix = context.active_object.evaluated_get(dg).matrix_world

        matrix = active_object_matrix @ matrix_from_axes(center, up, front)

        set_grid_transform(context, matrix, initial_matrix, interpolated = True)

        return {'FINISHED'}

class SetGridOrigin(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set"
    bl_label = "Set Grid Origin"
    bl_description = "Set grid origin from active object, face, vertex, or bone"

    bl_options = {'REGISTER', 'UNDO'}

    def get_enum_items(self, context):
        items = []

        items.append(('CURSOR', "(LEGACY) Cursor", "Align grid to 3D cursor", 'CURSOR', 0))
        items.append(('OBJECT', "(LEGACY) Object", "Align grid to the active object", 'OBJECT_DATA', 1))

        if SetGridOrigin.poll_face(context):
            items.extend([
                ('FACE', "(LEGACY) Face", "Align grid to the active face, with the active edge or vertex aligned to the grid; if no edge or vertex is active, the longest edge will be aligned to the grid", 'FACESEL', 3),
            ])

        if SetGridOrigin.poll_edge(context):
            items.append(('EDGE', "(LEGACY) Edge", "Translate grid to the active vertex", 'EDGESEL', 4))

        if SetGridOrigin.poll_vertex(context):
            items.append(('VERTEX', "(LEGACY) Vertex", "Translate grid to the active vertex", 'VERTEXSEL', 5))
            items.append(('VERTEX_PROJECT', "(LEGACY) Vertex Project", "Rotate the grid so that the vertex lies on a cardinal axis", 'VERTEXSEL', 6))

        if SetGridOrigin.poll_bone(context):
            items.append(('BONE', "(LEGACY) Bone", "Align grid to the active bone", 'BONE_DATA', 7))

        return items

    def get_enum(self):
        # This isn't great, but oh well.
        context = bpy.context

        if self.align_to_internal != -1:
            return self.align_to_internal

        default = 1

        if SetGridOrigin.poll_edge(context):
            default = 4

        if SetGridOrigin.poll_vertex(context):
            default = 5

        if SetGridOrigin.poll_face(context):
            default = 3

        if SetGridOrigin.poll_bone(context):
            default = 6


        self.align_to_internal = default
        

        return self.align_to_internal

    def set_enum(self, value):
        print("SET", value)
        self.align_to_internal = value

    # Internal, because there's no way to hide a sentinel item if I use `layout.prop`. (Well, there is, but it's very ugly -- and I won't speak of it again!)
    # Initially, I had used `prop_enum` to create an expanded button list, but that style doesn't work so well with so many options. (It couldn't be laid out horizontally, because the text started being truncated, and vertically it just looked ugly.)
    align_to_internal: bpy.props.IntProperty(default=-1, options={'SKIP_SAVE'})
    align_to: bpy.props.EnumProperty(
        name="Align to",
        items=get_enum_items,
        options={'SKIP_SAVE'},
        get=get_enum,
        set=set_enum,
        default=-1
    )

    # Skip save on these, because it's really annoying to have these save across different kinds.
    # The proper fix is probably to separate this operator into operators for face/vertex/edge/whatever.
    translation: bpy.props.BoolProperty(name="Translation", default=True, options={'SKIP_SAVE'})
    rotation: bpy.props.BoolProperty(name="Rotation", default=True, options={'SKIP_SAVE'})

    bone_head_tail: bpy.props.FloatProperty(name="Head/Tail", min=0.0, max=1.0, subtype='FACTOR')

    # The initial grid matrix, *including* the minimized roll portion.
    initial_grid_matrix: bpy.props.FloatVectorProperty(name="Initial Grid Matrix", subtype='MATRIX', size=(4, 4), options={'SKIP_SAVE'})
    initial_grid_origin_up: bpy.props.FloatVectorProperty(name="Initial Grid Origin Up", subtype='DIRECTION', size=3, options={'SKIP_SAVE'})

    @staticmethod
    def poll_object(context):
        return context.active_object is not None

    @staticmethod
    def poll_face(context):
        if context.mode != 'EDIT_MESH': return False

        data = context.active_object.data

        # Experimentally, the editmesh -> bmesh conversion is not the bottleneck. Hopefully this is fine!
        bm = bmesh.from_edit_mesh(data)
        active = bm.faces.active is not None and bm.faces.active.select
        bm.free()

        return active


    @staticmethod
    def poll_edge(context):
        if context.mode != 'EDIT_MESH': return False

        data = context.active_object.data

        bm = bmesh.from_edit_mesh(data)
        active = False

        if isinstance(bm.select_history.active, bmesh.types.BMEdge):
            active = True

        if len(bm.select_history) == 2 and isinstance(bm.select_history[0], bmesh.types.BMVert) and isinstance(bm.select_history[1], bmesh.types.BMVert):
            active = True

        bm.free()

        return active

    @staticmethod
    def poll_vertex(context):
        if context.mode != 'EDIT_MESH': return False

        data = context.active_object.data

        bm = bmesh.from_edit_mesh(data)
        active = isinstance(bm.select_history.active, bmesh.types.BMVert)
        bm.free()

        return active


    @staticmethod
    def poll_bone(context):
        # Assumption: context.active_pose_bone is `True` iff there's an active bone and the context is in pose mode.
        return (context.mode == 'EDIT_ARMATURE' or context.mode == 'POSE') and context.active_bone is not None

    def draw(self, context):
        layout = self.layout.column()
        layout.use_property_decorate = True
        layout.use_property_split = True

        align_to = layout.column(align = True)
        layout.prop(self, "align_to")

        rotation_supported = self.align_to != "VERTEX"

        layout.prop(self, "translation")

        rotation = layout.column()
        rotation.prop(self, "rotation")
        rotation.enabled = rotation_supported

        if self.poll_face(context):
            face = layout.column()

            face.enabled = (self.align_to == 'FACE_VERTEX' or self.align_to == 'FACE_EDGE') and self.rotation

        if self.poll_bone(context):
            bone = layout.column()
            bone.prop(self, "bone_head_tail")
            bone.enabled = self.align_to == 'BONE'

    def execute(self, context):
        if self.align_to == 'DEFAULT':
            match context.mode:
                case 'OBJECT':
                    self.align_to = 'OBJECT'
                case 'EDIT_MESH':
                    if self.poll_vertex(context):
                        self.align_to = 'VERTEX'
                    elif self.poll_edge(context):
                        self.align_to = 'EDGE'
                    else:
                        self.align_to = 'FACE_EDGE'
                case 'EDIT_ARMATURE' | 'POSE':
                    self.align_to = 'BONE'

        dg = context.evaluated_depsgraph_get()

        if not self.options.is_repeat:
            print("Initial rotation calculating")
            self.initial_grid_matrix = mathutils.Matrix()
            if context.scene.grid_origin is None:
                self.initial_grid_matrix = mathutils.Matrix()
                self.initial_grid_origin_up = mathutils.Vector((0, 0, 1))
            else:
                self.initial_grid_matrix = context.scene.evaluated_get(dg).grid_origin.matrix_world.inverted()
                self.initial_grid_origin_up = context.scene.grid_origin_up

        interpolated = not self.options.is_repeat

        clear_grid_transform(context, interpolated)

        dg.update()

        # The active object's matrix is required for every snap mode.
        active_object_matrix = context.active_object.evaluated_get(dg).matrix_world

        matrix = None

        rotation = self.rotation

        match self.align_to:
            case 'DEFAULT': assert False
            case 'OBJECT':
                matrix = active_object_matrix
            case 'FACE':
                data = context.active_object.data
                bm = bmesh.from_edit_mesh(data)
                face = bm.faces.active

                axis = None

                front_vector = None

                active_el = bm.select_history.active
                face_center = face.calc_center_median()

                if isinstance(active_el, bmesh.types.BMEdge):
                    front_vector = active_el.verts[1].co - active_el.verts[0].co
                elif isinstance(active_el, bmesh.types.BMVert):
                    front_vector = active_el.co - face_center

                if face is None:
                    matrix = active_object_matrix
                else:
                    origin, up, front = bmesh_face_axes(
                        face,
                        front_vector,
                    )
                    matrix = active_object_matrix @ matrix_from_axes(origin, up, front)

                bm.free()
            case 'EDGE':
                # An edge is WEIRD because there are two axes!
                data = context.active_object.data

                bm = bmesh.from_edit_mesh(data)

                vertices = []
                if isinstance(edge := bm.select_history.active, bmesh.types.BMEdge):
                    vertices = [*edge.verts]
                else:
                    vertices = [*bm.select_history]

                center = (vertices[0].co + vertices[1].co) / 2
                forwards = vertices[1].co - vertices[0].co

                center = active_object_matrix @ center
                forwards = active_object_matrix.to_3x3() @ forwards

                initial_rotation = self.initial_grid_matrix.to_quaternion()

                up = self.initial_grid_origin_up
                matrix = matrix_from_axes(center, up, forwards)
            case 'VERTEX':
                data = context.active_object.data
                bm = bmesh.from_edit_mesh(data)
                vertex = bm.select_history.active

                rotation = False

                if vertex is None:
                    matrix = active_object_matrix
                else:
                    matrix = active_object_matrix @ mathutils.Matrix.Translation(vertex.co)
            case 'VERTEX_PROJECT':
                data = context.active_object.data
                bm = bmesh.from_edit_mesh(data)
                vertex = bm.select_history.active

                center = active_object_matrix @ vertex.co
                axis = center - self.initial_grid_matrix.to_translation()

                up = self.initial_grid_origin_up

                print("vp", center, axis, up)

                matrix = matrix_from_axes(center, up, axis)
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

                # bone_matrix = mathutils.Matrix.Translation((0, 0, 0))

                matrix = active_object_matrix @ bone_matrix
            case 'CURSOR':
                matrix = context.scene.cursor.matrix.copy()

        assert matrix is not None

        if not self.translation:
            matrix = mathutils.Matrix.Translation(self.initial_grid_matrix.to_translation()) @ matrix.to_quaternion().to_matrix().to_4x4()

        if not rotation:
            translation = matrix.to_translation()
            matrix = mathutils.Matrix.Translation(translation) @ self.initial_grid_matrix.to_quaternion().to_matrix().to_4x4()

        print(self.options.is_repeat)
        set_grid_transform(context, matrix, self.initial_grid_matrix, interpolated)

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
        history_pre_matrix = scene.grid_origin.matrix_world.copy()
    else:
        history_pre_matrix = mathutils.Matrix()

@persistent
def history_post_handler(scene):
    global history_pre_matrix

    print("hist post", scene.grid_origin)
    
    if scene.grid_origin:
        matrix = scene.grid_origin.matrix_world
    else:
        matrix = mathutils.Matrix()

    apply_matrix_to_misc_view(bpy.context, history_pre_matrix @ matrix.inverted(), interpolated = False)

def menu_func(self, context):
    self.layout.separator()
    # self.layout.operator(SetGridOrigin.bl_idname)
    self.layout.operator(ClearGridOrigin.bl_idname)
    self.layout.operator(SetGridOriginFromActive.bl_idname)
    self.layout.operator(SetGridOriginFromCursor.bl_idname)
    self.layout.operator(SetGridOriginFromVertices.bl_idname)

class VIEW3D_MT_local_grid_pie(bpy.types.Menu):
    bl_label = "Local Grid"

    def draw(self, _context):
        layout = self.layout

        pie = layout.menu_pie()
        pie.operator(ClearGridOrigin.bl_idname)
        pie.operator(SetGridOriginFromActive.bl_idname)
        pie.operator(SetGridOriginFromCursor.bl_idname)
        pie.operator(SetGridOriginFromVertices.bl_idname)
        # pie.operator_enum(SetGridOrigin.bl_idname, "align_to")

addon_keymaps = []

classes = [
    GridSnapAddonPreferences,
    SetGridOrigin,
    SetGridOriginFromActive,
    SetGridOriginFromCursor,
    SetGridOriginFromVertices,
    ClearGridOrigin,

    VIEW3D_MT_local_grid_pie,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    wm = bpy.context.window_manager
    if wm.keyconfigs.addon:
        # Origin/Pivot
        km = wm.keyconfigs.addon.keymaps.new(name="3D View Generic", space_type='VIEW_3D')
        kmi = km.keymap_items.new('wm.call_menu_pie', 'Y', 'PRESS', ctrl=True)
        kmi.properties.name = 'VIEW3D_MT_local_grid_pie'
        addon_keymaps.append((km, kmi))

    bpy.types.VIEW3D_MT_view.append(menu_func)

    bpy.app.handlers.undo_pre.append(history_pre_handler)
    bpy.app.handlers.undo_post.append(history_post_handler)

    bpy.app.handlers.redo_pre.append(history_pre_handler)
    bpy.app.handlers.redo_post.append(history_post_handler)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        for km, kmi in addon_keymaps:
            km.keymap_items.remove(kmi)
    addon_keymaps.clear()

    bpy.types.VIEW3D_MT_view.remove(menu_func)

    bpy.app.handlers.undo_pre.remove(history_pre_handler)
    bpy.app.handlers.undo_post.remove(history_post_handler)

    bpy.app.handlers.redo_pre.remove(history_pre_handler)
    bpy.app.handlers.redo_post.remove(history_post_handler)
