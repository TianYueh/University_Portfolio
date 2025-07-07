#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;
//Input/Output part is important, so be sure to check how everything works

in VS_OUT {
	vec3 normal;
	vec3 fragpos;
	vec2 texCoord;
} gs_in[];

uniform mat4 P;
uniform vec3 windshift;


out vec3 fragposGS;
out vec3 normalGS;
out vec2 texCoordGS;
out vec3 colorGS;

//out vec3 color;

//const float FUR_LENGTH = 0.065;
uniform float FUR_LENGTH;
uniform float time;

//This is the most important part we want you to implement on yourself
//Use demo code for a learning example and design a geometry shader for your object

void main()
{
    for (int i = 0; i < gl_in.length(); ++i) {

	

		fragposGS = gs_in[i].fragpos;
        normalGS = normalize(gs_in[i].normal);
        texCoordGS = gs_in[i].texCoord;

        // Flashing effect using a sine function
        float flashIntensity = 0.5 + 0.3 * sin(time);  // Adjust the frequency and amplitude as needed
        colorGS = vec3(flashIntensity, flashIntensity, flashIntensity);
        if(FUR_LENGTH > 0.0)
		{
			colorGS = vec3(1.0, 1.0, 1.0);
		}


        
        
        gl_Position = P * gl_in[i].gl_Position + vec4(gs_in[i].normal, 0.0) * FUR_LENGTH;
        EmitVertex();
		}
	EndPrimitive();
  
}