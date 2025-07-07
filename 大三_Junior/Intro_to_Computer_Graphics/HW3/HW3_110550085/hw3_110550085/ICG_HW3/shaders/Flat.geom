#version 330 core

// TODO:
// implement Flat shading

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec2 texCoord[];
in vec4 worldPos[];
in vec3 normal[];

out vec2 fragTexCoord;
out vec3 fragNormal;
out vec4 fragWorldPos;

uniform vec3 lightPos;

void main()
{
    vec3 faceNormal = cross((worldPos[1] - worldPos[0]).xyz, (worldPos[2] - worldPos[0]).xyz);
    faceNormal = normalize(faceNormal);
    for (int i = 0; i < gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position;
        fragTexCoord = texCoord[i];
        fragWorldPos = worldPos[i];
        fragNormal = faceNormal;
        EmitVertex();
    }
    EndPrimitive();
}
