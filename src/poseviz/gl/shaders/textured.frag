#version 330 core

in vec2 v_texcoord;

uniform sampler2D u_texture;
uniform float u_opacity;

out vec4 frag_color;

void main() {
    vec4 tex_color = texture(u_texture, v_texcoord);
    frag_color = vec4(tex_color.rgb, u_opacity);
}
