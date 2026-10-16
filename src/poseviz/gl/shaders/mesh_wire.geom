#version 330 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 v_normal[];

out vec3 g_normal;
// Screen-linear (noperspective) barycentrics: with fwidth() in the fragment
// shader this gives edges of constant width in pixels.
noperspective out vec3 g_bary;

void main() {
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        g_normal = v_normal[i];
        g_bary = vec3(i == 0 ? 1.0 : 0.0, i == 1 ? 1.0 : 0.0, i == 2 ? 1.0 : 0.0);
        EmitVertex();
    }
    EndPrimitive();
}
