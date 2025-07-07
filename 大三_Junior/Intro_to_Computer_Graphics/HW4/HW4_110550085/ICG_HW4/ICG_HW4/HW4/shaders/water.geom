#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
} gs_in[];

out GS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 Color;
} gs_out;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;
uniform float time;  // Time variable for animation

void main()
{
    for (int i = 0; i < 3; ++i)
    {
        vec4 fragPos = M * vec4(gs_in[i].FragPos, 1.0);
        float displacement = sin(fragPos.x * 5.0 + fragPos.z * 5.0 + time) * 0.1;
        vec4 offset = vec4(0.0, displacement, 0.0, 0.0);
        
        gl_Position = P * V * (fragPos + offset);
        gs_out.FragPos = vec3(M * vec4(gs_in[i].FragPos, 1.0));
        gs_out.Normal = mat3(transpose(inverse(M))) * gs_in[i].Normal;
        gs_out.TexCoords = gs_in[i].TexCoords;
        gs_out.Color = vec3(0.0, 0.5, 1.0); // Blue color for water
        EmitVertex();
    }
    EndPrimitive();
}
