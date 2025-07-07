#version 330 core

in vec2 texCoord;
in vec4 ambient;
in vec4 diffuse;
in vec4 specular;
uniform sampler2D deerTexture;
out vec4 fragColor;

void main()
{
	vec4 objectColor = texture2D(deerTexture, texCoord);
	fragColor = ambient * objectColor + diffuse * objectColor + specular;
}