#version 450

layout(location = 0) in vec4 position;
layout(location = 1) in vec2 texCoords;

layout(location = 0) out vec2 out_texcoords;

void main(void) {
    out_texcoords = texCoords;
    gl_Position = position;
}