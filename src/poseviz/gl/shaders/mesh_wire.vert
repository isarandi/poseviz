#version 330 core

// Uniform-colour mesh with edges drawn in the fragment shader ("solid
// wireframe"): the geometry stage stamps barycentric coordinates on each
// triangle, so faces and edges are rasterised together, in draw order.

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_view_proj;

out vec3 v_normal;

void main() {
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    v_normal = in_normal;
}
