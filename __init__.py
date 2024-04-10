# Local Grid
# Copyright (C) 2024 LiterallyVoid
# literallyvoid@gmail.com
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

bl_info = {
    "name": "Local Grid",
    "author": "LiterallyVoid",
    "version": (1, 0),
    "blender": (4, 1, 0),
    "location": "View3D > View",
    "description": "Local Grid",
    "warning": "",
    "doc_url": "",
    "tracker_url": "",
    "category": "3D View",
}

import bpy, mathutils, bmesh, math
from typing import Optional
from bpy.app.handlers import persistent

grid_origin_empty_name = "Local Grid: Grid Origin Inverse"

def create_transformed_empty(context, matrix: mathutils.Matrix):
    try:
        empty = bpy.data.objects[grid_origin_empty_name]
        if empty.users > 0:
            raise ValueError
    except:
        empty = bpy.data.objects.new(name = grid_origin_empty_name, object_data = None)

    empty.matrix_world = matrix.inverted()

    if empty.parent:
        # This maybe happened in testing. I'm almost positive that it was because of me changing the addon, but if this ever happens then it will be very bad.
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

def apply_matrix_to_area(context, interpolated, window, screen, area, translation, rotation):
    try:
        space = area.spaces[0]
    except:
        # ???
        return

    if space.type != 'VIEW_3D':
        return

    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences

    rotation_matrix = rotation.to_matrix()

    for region in area.regions:
        if region.type != 'WINDOW':
            continue

        if not region.data:
            continue

        if region.data.view_perspective == 'CAMERA':
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
            with context.temp_override(window = window, screen = screen, area = area, view = space, region = region):
                bpy.ops.view3d.view_roll('INVOKE_REGION_WIN', angle = -roll)
        else:
            region.data.view_rotation = tracked

    

def apply_matrix_to_misc_view(context, matrix, interpolated = True):
    preferences = context.preferences
    addon_prefs = preferences.addons[__name__].preferences

    matrix = matrix.inverted()

    translation, rotation, _ = matrix.decompose()

    # Correct the cursor
    context.scene.cursor.matrix = matrix @ context.scene.cursor.matrix

    # Correct every 3D view in the active workspace
    for window in context.window_manager.windows:
        if window.scene != context.scene:
            continue

        for area in window.screen.areas:
            apply_matrix_to_area(
                context,
                interpolated,
                window,
                window.screen,
                area,
                translation,
                rotation,
            )

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

    candidates = [
        mathutils.Quaternion(),
        mathutils.Quaternion((1, 0, 0), math.pi * 0.5),
        mathutils.Quaternion((1, 0, 0), math.pi),
        mathutils.Quaternion((1, 0, 0), math.pi * 1.5),
        mathutils.Quaternion((0, 1, 0), math.pi * 0.5),
        mathutils.Quaternion((0, 1, 0), math.pi * 1.5),
    ]

    for yaw in range(1, 4):
        for candidate in candidates[0:6]:
            candidates.append(candidate @ mathutils.Quaternion((0, 0, 1), math.pi * 0.5 * yaw))

    # Rotations: [
    #    original, +y, -z, -y, +x, -x, (no yaw)
    #    original, +y, -z, -y, +x, -x, (all yawed 90°)
    #    original, +y, -z, -y, +x, -x, (all yawed 180°)
    #    original, +y, -z, -y, +x, -x, (all yawed 270°)
    # so prioritize index 0, then 6n, then the rest.

    scored = []

    # @TODO: max(..., key = ...)?
    for i, candidate in enumerate(candidates):
        total_rotation = rotation @ candidate

        _axis, angle = total_rotation.to_axis_angle()
        score = angle

        # Prefer no orientation change.
        if i == 0: score -= 0.02

        # Prefer only yaw.
        elif i % 6 == 0: score -= 0.01

        scored.append((matrix @ candidate.to_matrix().to_4x4(), score))

    return min(scored, key = lambda matrix_score: matrix_score[1])[0]
    # scored.sort(key = lambda tup: tup[1])

    # return scored[-1][0]

# `previous_matrix` will be used to minimize roll.
def set_grid_transform(context, transform: mathutils.Matrix, previous_matrix: Optional[mathutils.Matrix] = None, interpolated = True, *, move_cursor_to_origin = True):
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

    if move_cursor_to_origin and addon_prefs.move_cursor_to_origin:
        context.scene.cursor.matrix = mathutils.Matrix()

bpy.types.Scene.grid_origin = bpy.props.PointerProperty(type=bpy.types.Object, name="Grid Origin", description="The Empty currently set as the Grid Origin. Its transform is the inverse transform of the current grid transform", options=set())
bpy.types.Scene.grid_origin_up = bpy.props.FloatVectorProperty(name="Grid +Z Axis, in world space", description="May differ from the current grid's Z axis if Minimize Roll is set. Used where the up axis is an important distinction, as in 'Project Cursor' and 'Align Edge'", subtype='DIRECTION')
bpy.types.Scene.grid_origin_side = bpy.props.FloatVectorProperty(name="Some vector perpendicular to `grid_origin_up`, in world space", description="May differ from the current grid transform if Minimize Roll is set", subtype='DIRECTION')

# Return the global `up` vector.
# May differ from the current grid transform's Z axis if a different Z axis was chosen to minimize roll.
def get_grid_up(scene):
    if scene.grid_origin is not None:
        return scene.grid_origin_up

    return mathutils.Vector((0, 0, 1))

class GridSnapAddonPreferences(bpy.types.AddonPreferences):
    bl_idname = __name__

    reset_roll: bpy.props.BoolProperty(
        name="Reset Camera Roll",
        description="Roll the view so that it points up whenever the grid is changed. This makes navigation easier if your orbit method is Turntable; otherwise it can be disorienting",
        default=True
    )

    minimize_roll: bpy.props.BoolProperty(
        name="Minimize Roll",
        description="Instead of always aligning the +Z axis to the object or face, allow choosing a different Z axis to minimize the angle to the world's coordinate space",
        default=True
    )

    move_cursor_to_origin: bpy.props.BoolProperty(
        name="Snap Cursor",
        description="Snap the cursor to the grid origin whenever the grid is changed",
        default=True
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "reset_roll")
        layout.prop(self, "minimize_roll")
        layout.prop(self, "move_cursor_to_origin")

# Choose a cardinal direction that's maximally perpendicular to `orthogonal_to`.
def choose_orthogonal_axis(orthogonal_to):
    orthogonal_to = orthogonal_to.normalized()

    axes = [
        mathutils.Vector((1, 0, 0)),
        mathutils.Vector((0, 1, 0)),
        mathutils.Vector((0, 0, 1)),
    ]

    return min(axes, key = lambda axis: abs(axis.dot(orthogonal_to)))


# Create a matrix from an origin and two axes, but where `up_vector` and `front_vector` are not perpendicular, create a matrix where `up_vector` is aligned to an axis.
def matrix_from_axes_prefer_up(
    origin: mathutils.Vector,
    up_vector: mathutils.Vector,
    front_vector: mathutils.Vector,
) -> mathutils.Matrix:
    up_vector = up_vector.normalized()

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

# Create a matrix from an origin and two vectors, but where `up_vector` and `front_vector` are not perpendicular, create a matrix where `front_vector` is aligned to an axis.
def matrix_from_axes_prefer_front(
    origin: mathutils.Vector,
    up_vector: mathutils.Vector,
    front_vector: mathutils.Vector,
) -> mathutils.Matrix:
    front_vector = front_vector.normalized()

    right = up_vector.cross(front_vector).normalized()
    up = front_vector.cross(right).normalized()

    back = -front_vector

    mat = mathutils.Matrix((
        mathutils.Vector((*right, 0)),
        mathutils.Vector((*back, 0)),
        mathutils.Vector((*up, 0)),
        origin.to_4d(),
    )).transposed()

    return mat

# If no `front_vector` is given, choose the longest edge.
# This is an arbitrary deterministic choice.
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

class Cursor(bpy.types.Operator):
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


class TranslateToSelected(bpy.types.Operator):
    bl_idname = "view3d.grid_translate_to_selected"
    bl_label = "Translate Grid to Selected"
    bl_description = "Center grid to the middle of selected item(s), ignoring rotation"

    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.ops.view3d.snap_cursor_to_selected.poll()

    def execute(self, context):
        dg = context.evaluated_depsgraph_get()

        initial_matrix = clear_grid_transform(context, False)

        # Just use `view3d.snap_cursor_to_selected` and revert the cursor later.
        cursor_matrix = context.scene.cursor.matrix

        bpy.ops.view3d.snap_cursor_to_selected()

        matrix = context.scene.cursor.matrix.copy()

        # Revert cursor matrix.
        context.scene.cursor.matrix = cursor_matrix

        translation = matrix.to_translation()
        matrix = mathutils.Matrix.Translation(translation) @ initial_matrix.to_quaternion().to_matrix().to_4x4()

        set_grid_transform(context, matrix, initial_matrix)

        return {'FINISHED'}


class ProjectCursor(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_project_cursor"
    bl_label = "Project Local Grid to Cursor"
    bl_description = "Rotate grid around its current origin so that the 3D Cursor lies on a cardinal axis"

    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        dg = context.evaluated_depsgraph_get()


        if context.scene.grid_origin is None:
            up = mathutils.Vector((0, 0, 1))
        else:
            up = context.scene.evaluated_get(dg).grid_origin_up


        initial_matrix = clear_grid_transform(context, False)

        dg = context.evaluated_depsgraph_get()

        cursor_matrix = context.scene.evaluated_get(dg).cursor.matrix

        center = initial_matrix @ mathutils.Vector((0, 0, 0))
        front = cursor_matrix.to_translation() - center

        matrix = matrix_from_axes_prefer_front(center, up, front)


        set_grid_transform(context, matrix, initial_matrix, move_cursor_to_origin = False)


        return {'FINISHED'}


class ProjectSelected(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_project_selected"
    bl_label = "Project Local Grid to Vertex"
    bl_description = "Rotate grid around its current origin so that the middle of selected item(s) is on a cardinal axis"

    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return bpy.ops.view3d.snap_cursor_to_selected.poll()

    def execute(self, context):
        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        up = get_grid_up(context.scene)
        initial_matrix = clear_grid_transform(context, False)


        # Just use `view3d.snap_cursor_to_selected` and revert the cursor later.
        cursor_matrix = context.scene.cursor.matrix

        bpy.ops.view3d.snap_cursor_to_selected()

        front = context.scene.cursor.matrix.to_translation()

        # Revert cursor matrix.
        context.scene.cursor.matrix = cursor_matrix

        center = initial_matrix @ mathutils.Vector((0, 0, 0))
        front = front - center

        matrix = matrix_from_axes_prefer_front(center, up, front)


        set_grid_transform(context, matrix, initial_matrix, move_cursor_to_origin = False)


        return {'FINISHED'}


class Active(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_set_active"
    bl_label = "Set Local Grid from Active"
    bl_description = "Orient and center the grid around the active item"

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
        initial_matrix = clear_grid_transform(context, interpolated = True)
        initial_rotation = initial_matrix.to_quaternion().to_matrix().to_4x4()

        dg = context.evaluated_depsgraph_get()

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

                matrix = active_object_matrix @ matrix_from_axes_prefer_up(origin, up, front)


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

                matrix = matrix_from_axes_prefer_up(origin, up, front)


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

                #bone_matrix = context.active_bone.matrix_local
                #bone_length = context.active_bone.length

                raise ValueError("this should not happen") # If this happens, maybe removing this `raise` and uncommenting the above lines will help?

            # There used to be a head/tail slider, but I didn't want to add one-off parameters to a multipurpose operator.
            #bone_matrix = bone_matrix @ mathutils.Matrix.Translation((0, bone_length * self.bone_head_tail, 0))

            matrix = active_object_matrix @ bone_matrix


        else:
            matrix = active_object_matrix

            # Blender runs on an assumption of a Z-up, Y-forwards space, but camera matrices are Z-backwards and Y-up.
            # Rotate camera matrices so that moving things up/down still works.
            if context.active_object.type == 'CAMERA':
                matrix @= mathutils.Quaternion((1, 0, 0), -math.pi * 0.5).to_matrix().to_4x4()

        set_grid_transform(context, matrix, initial_matrix, interpolated = True)

        return {'FINISHED'}

class TriangulateVertices(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_triangulate_vertices"
    bl_label = "Triangulate Grid Origin From Vertices"
    bl_description = "Set grid origin from three selected vertices, placing the origin at the third vertex and any cardinal axis on the line between the second and third"

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
            raise ValueError("selected vertices are colinear")

        initial_matrix = clear_grid_transform(context, interpolated = True)

        dg = context.evaluated_depsgraph_get()

        active_object_matrix = context.active_object.evaluated_get(dg).matrix_world

        matrix = active_object_matrix @ matrix_from_axes_prefer_up(center, up, front)

        set_grid_transform(context, matrix, initial_matrix, interpolated = True)

        return {'FINISHED'}


class AlignToEdge(bpy.types.Operator):
    bl_idname = "view3d.grid_origin_align_to_edge"
    bl_label = "Align Grid Origin to Edge"
    bl_description = "Rotate grid around its current origin so that active edge is aligned to an axis"

    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if context.mode != 'EDIT_MESH': return False
        if context.active_object is None: return False

        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        # If there are two selected vertices, choose the line between them
        if len(bm.select_history) == 2 and \
            all([isinstance(elem, bmesh.types.BMVert) for elem in bm.select_history]):
            return True

        # If the active element is an edge, choose it
        if isinstance(bm.select_history.active, bmesh.types.BMEdge):
            return True

        return False

    def execute(self, context):
        data = context.active_object.data
        bm = bmesh.from_edit_mesh(data)

        vector = None

        # If there are two selected vertices, choose the line between them
        if len(bm.select_history) == 2 and \
            all([isinstance(elem, bmesh.types.BMVert) for elem in bm.select_history]):
            vector = bm.select_history[1].co - bm.select_history[0].co


        # If the active element is an edge, choose it
        elif isinstance(bm.select_history.active, bmesh.types.BMEdge):
            edge = bm.select_history.active
            vector = edge.verts[1].co - edge.verts[0].co

        else:
            raise ValueError("AlignToEdge: poll should have failed")

        up = get_grid_up(context.scene)
        initial_matrix = clear_grid_transform(context, interpolated = True)

        dg = context.evaluated_depsgraph_get()

        center = initial_matrix @ mathutils.Vector((0, 0, 0))

        active_object_matrix = context.active_object.evaluated_get(dg).matrix_world

        vector = active_object_matrix.to_3x3() @ vector

        matrix = matrix_from_axes_prefer_front(center, up, vector)

        set_grid_transform(context, matrix, initial_matrix, interpolated = True)

        return {'FINISHED'}


class ClearGridOrigin(bpy.types.Operator):
    bl_idname = "view3d.reset_local_grid"
    bl_label = "Reset Local Grid"
    bl_description = "Reset grid back to the scene's origin"

    # It feels a lot better if this operator's always available.
    # It's idempotent anyway.

    # @classmethod
    # def poll(cls, context):
    #     return context.scene.grid_origin is not None

    def execute(self, context):
        if context.scene.grid_origin is None:
            # Don't add an undo entry if nothing happens.
            return {'CANCELLED'}

        clear_grid_transform(context)

        return {'FINISHED'}

history_pre_matrix = None

@persistent
def history_pre_handler(scene):
    global history_pre_matrix

    if scene.grid_origin:
        history_pre_matrix = scene.grid_origin.matrix_world.copy()
    else:
        history_pre_matrix = None

@persistent
def history_post_handler(scene):
    global history_pre_matrix

    print("hist post", scene.grid_origin)
    
    if scene.grid_origin:
        matrix = scene.grid_origin.matrix_world
    else:
        matrix = None

    if matrix is None and history_pre_matrix is None:
        return

    combined = \
        (history_pre_matrix or mathutils.Matrix()) \
      @ (matrix or mathutils.Matrix()).inverted()

    apply_matrix_to_misc_view(bpy.context, combined, interpolated = False)

def menu_func(self, context):
    self.layout.menu("VIEW3D_MT_local_grid", text="Local Grid")

class VIEW3D_MT_local_grid(bpy.types.Menu):
    bl_label = "Local Grid"

    def draw(self, _context):
        layout = self.layout
        layout.operator(ClearGridOrigin.bl_idname)
        layout.operator(Active.bl_idname)
        layout.operator(TriangulateVertices.bl_idname)
        layout.operator(ProjectSelected.bl_idname)
        layout.operator(AlignToEdge.bl_idname)

        layout.operator(TranslateToSelected.bl_idname)
        layout.operator(Cursor.bl_idname)
        layout.operator(ProjectCursor.bl_idname)

class VIEW3D_MT_local_grid_pie(bpy.types.Menu):
    bl_label = "Local Grid"

    def draw(self, _context):
        layout = self.layout

        pie = layout.menu_pie()

        items = [
            (TranslateToSelected, "Selected", 'RESTRICT_SELECT_OFF'),

            (TriangulateVertices, "Triangulate Vertices", 'VERTEXSEL'),
            (Cursor, "Cursor", 'CURSOR'),

            (ClearGridOrigin, "Reset", 'WORLD'),
            (Active, "Active", 'PIVOT_ACTIVE'),

            (AlignToEdge, "Align Edge", 'EDGESEL'),
            (ProjectCursor, "Project Cursor", 'CURSOR'),

            (ProjectSelected, "Project Selected", 'RESTRICT_SELECT_OFF'),
        ]

        order = [
            "W", "E", "S", "N", "NW", "NE", "SW", "SE",
        ]

        i = 0

        for clazz, text, icon in [
            # West, East
            items[3], items[4],

            # South, North
            items[7], items[0],

            # Northwest, Northeast
            items[1], items[2],

            # Southwest, Southeast
            items[5], items[6],
        ]:
            pie.operator(clazz.bl_idname, text=text, icon=icon)



addon_keymaps = []

classes = [
    GridSnapAddonPreferences,

    ClearGridOrigin,

    Active,
    TriangulateVertices,
    AlignToEdge,

    ProjectSelected,
    TranslateToSelected,

    Cursor,
    ProjectCursor,

    VIEW3D_MT_local_grid,
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
