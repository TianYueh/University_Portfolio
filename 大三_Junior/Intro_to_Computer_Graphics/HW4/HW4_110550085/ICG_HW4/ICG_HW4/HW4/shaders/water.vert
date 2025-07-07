#version 330 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoords;

out VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} vs_out;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

void main()
{
    gl_Position = P * V * M * vec4(aPos, 1.0);
    vs_out.FragPos = vec3(M * vec4(aPos, 1.0));
    vs_out.Normal = mat3(transpose(inverse(M))) * aNormal;
    vs_out.TexCoords = aTexCoords;
}
