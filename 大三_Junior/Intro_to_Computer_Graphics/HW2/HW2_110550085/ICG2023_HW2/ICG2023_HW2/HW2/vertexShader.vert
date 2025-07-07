#version 330 core

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

uniform float squeezeFactor;
uniform float offset;

out vec2 texCoord;
out vec3 normal;

vec4 worldPos;

void main()
{
	// TODO: Implement squeeze effect and update normal transformation
	//   1. Adjust the vertex position to create a squeeze effect based on squeezeFactor.
	//   2. Update worldPos using the model matrix (M).
	//   3. Calculate the final gl_Position using the perspective (P), view (V), and updated worldPos.
	//   4. Update the normal transformation for lighting calculations.
	// Note: Ensure to handle the squeeze effect for both y and z coordinates.
	vec3 squeezedPos = aPos;
	squeezedPos.y += aPos.z * sin(squeezeFactor) / 2.0;  // Squeeze effect for y coordinate
    squeezedPos.z += aPos.y * sin(squeezeFactor) / 2.0;  // Squeeze effect for z coordinate
    //squeezedPos.y *= squeezeFactor;
    //squeezedPos.z *= squeezeFactor;

	worldPos = M * vec4(squeezedPos, 1);

	gl_Position = P * V * worldPos;
	mat4 normal_transform = transpose(inverse(M));
	normal = normalize((normal_transform * vec4(aNormal, 0.0)).xyz);

	texCoord = aTexCoord;
}