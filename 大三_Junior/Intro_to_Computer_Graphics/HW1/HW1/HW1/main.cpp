#include <iostream>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Object.h"
#include "Shader.h"

using namespace std;

void framebufferSizeCallback(GLFWwindow* window, int width, int height);
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
unsigned int modelVAO(Object& model);
void drawModel(const string& target, unsigned int& shaderProgram, const glm::mat4& M, const glm::mat4& V, const glm::mat4& P);
//Added Parameter
float rotationTimes = 1.0f; // Adjust this value as needed
float rotateAngle = 0.0f;
float minuteHandAngle = 0.0f;
float hourHandAngle = 0.0f;
float rabbitAngle = 0.0f;
float tortoiseAngle = 0.0f;

// Objects to display
Object rectangleModel("obj/rectangle.obj");
Object triangleModel("obj/triangle.obj");
Object clockModel("obj/clock.obj");
Object clockHandModel("obj/clock_hand.obj");
Object rabbitModel("obj/rabbit.obj");
Object tortoiseModel("obj/tortoise.obj");

unsigned int rectangleVAO, triangleVAO, clockVAO, clockHandVAO, rabbitVAO, tortoiseVAO;
int windowWidth = 800, windowHeight = 600;

bool isKey3Pressed = false;

int main()
{
	// Initialization
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFE_OPENGL_FORWARD_COMPACT, GL_TRUE);
#endif

	GLFWwindow* window = glfwCreateWindow(windowWidth, windowHeight, "HW1", NULL, NULL);
	if (window == NULL)
	{
		cout << "Failed to create GLFW window\n";
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
	glfwSetKeyCallback(window, keyCallback);
	glfwSwapInterval(1);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		cout << "Failed to initialize GLAD\n";
		return -1;
	}

	// VAO, VBO
	rectangleVAO = modelVAO(rectangleModel);
	triangleVAO = modelVAO(triangleModel);
	clockVAO = modelVAO(clockModel);
	clockHandVAO = modelVAO(clockHandModel);
	rabbitVAO = modelVAO(rabbitModel);
	tortoiseVAO = modelVAO(tortoiseModel);

	// Shaders
	Shader shader("vertexShader.vert", "fragmentShader.frag");
	glUseProgram(shader.program);

	// TODO: Enable depth test, face culling
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glFrontFace(GL_CCW);


	// Display loop
	glfwGetFramebufferSize(window, &windowWidth, &windowHeight);
	glViewport(0, 0, windowWidth, windowHeight);

	double lastTime = glfwGetTime();
	int numFrames = 0;
	

	while (!glfwWindowShouldClose(window))
	{
		// Calculate time per frame
		double currentTime = glfwGetTime();
		numFrames++;
		minuteHandAngle += 1.0f *rotationTimes;
		hourHandAngle += 1.0f / 60.0f *rotationTimes;
		rabbitAngle -= 0.7f *rotationTimes;
		tortoiseAngle -= 0.35f *rotationTimes;
		// If last cout was more than 1 sec ago
		if (currentTime - lastTime >= 1.0)
		{
			// Print and reset timer
			cout << 1000.0 / numFrames << " ms/frame\n";
			numFrames = 0;
			lastTime += 1.0;
		}

		glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// TODO: Create model, view, and perspective matrix

		glm::mat4 modelMatrix = glm::mat4(1.0f);  // Initialize with an identity matrix
		glm::mat4 viewMatrix = glm::lookAt(glm::vec3(0.0f, 30.0f, 50.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 projectionMatrix = glm::perspective(glm::radians(45.0f), (float)windowWidth / (float)windowHeight, 0.1f, 100.0f);



		// TODO: Draw base of clock tower
		glm::mat4 groundModelMatrix = glm::translate(modelMatrix, glm::vec3(0.0f, -10.0f, -3.0f));
		glm::mat4 f_groundModelMatrix = glm::scale(groundModelMatrix, glm::vec3(20.0f, 1.0f, 21.0f));
		drawModel("rectangle", shader.program, f_groundModelMatrix, viewMatrix, projectionMatrix);


		
		// TODO: Draw body of clock tower
		glm::mat4 towerModelMatrix = glm::translate(groundModelMatrix, glm::vec3(0.0f, 15.0f, 3.0f));
		towerModelMatrix = glm::rotate(towerModelMatrix, glm::radians(rotateAngle), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 f_towerModelMatrix = glm::scale(towerModelMatrix, glm::vec3(4.5f, 10.0f, 3.5f));
		drawModel("rectangle", shader.program, f_towerModelMatrix, viewMatrix, projectionMatrix);

		

		
		// TODO: Draw roof of clock tower
		//glm::mat4 roofModelMatrix = glm::mat4(1.0f);
		//roofModelMatrix = glm::translate(towerModelMatrix, glm::vec3(0.0f, 5.0f, 0.0f));
		glm::mat4 roofModelMatrix = glm::translate(towerModelMatrix, glm::vec3(-0.2f, 11.25f, -0.35f));
		roofModelMatrix = glm::scale(roofModelMatrix, glm::vec3(5.0f, 4.0f, 3.3f));
		drawModel("triangle", shader.program, roofModelMatrix, viewMatrix, projectionMatrix);

		
		// TODO: Draw clock on the clock tower
		glm::mat4 clockModelMatrix = glm::translate(towerModelMatrix, glm::vec3(0.0f, 4.5f, 4.3f));
		glm::mat4 f_clockModelMatrix = glm::scale(clockModelMatrix, glm::vec3(0.013f, 0.013f, 0.013f));
		f_clockModelMatrix = glm::rotate(f_clockModelMatrix, glm::radians(90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		drawModel("clock", shader.program, f_clockModelMatrix, viewMatrix, projectionMatrix);

		
		// TODO: Draw minute hand on the clock 
		//glm::mat4 minuteHandModelMatrix = glm::mat4(1.0f);
		glm::mat4 minuteHandModelMatrix = glm::translate(clockModelMatrix, glm::vec3(0.0f, 0.0f, 0.6f));
		minuteHandModelMatrix = glm::rotate(minuteHandModelMatrix, glm::radians(-180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		minuteHandModelMatrix = glm::rotate(minuteHandModelMatrix, glm::radians(minuteHandAngle), glm::vec3(0.0f, 0.0f, 1.0f));
		minuteHandModelMatrix = glm::scale(minuteHandModelMatrix, glm::vec3(0.8f, 0.7f, 1.0f));
		drawModel("clock hand", shader.program, minuteHandModelMatrix, viewMatrix, projectionMatrix);

		
		// TODO: Draw hour hand on the clock
		glm::mat4 hourHandModelMatrix = glm::translate(clockModelMatrix, glm::vec3(0.0f, 0.0f, 0.25f));
		hourHandModelMatrix = glm::rotate(hourHandModelMatrix, glm::radians(-180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		hourHandModelMatrix = glm::rotate(hourHandModelMatrix, glm::radians(hourHandAngle), glm::vec3(0.0f, 0.0f, 1.0f));
		hourHandModelMatrix = glm::scale(hourHandModelMatrix, glm::vec3(1.0f, 0.6f, 0.6f));
		drawModel("clock hand", shader.program, hourHandModelMatrix, viewMatrix, projectionMatrix);

		

		// TODO: Draw rabbit revolves around the clock tower (not (0, 0, 0))
		glm::mat4 rabbitModelMatrix = glm::mat4(1.0f);
		rabbitModelMatrix = glm::translate(groundModelMatrix, glm::vec3(15.0f, 1.0f, 0.0f));
		rabbitModelMatrix = glm::scale(rabbitModelMatrix, glm::vec3(0.08f, 0.08f, 0.08f));
		// Implement rotation for the rabbit as it revolves around the clock tower

		glm::vec3 axis(0.0f, 1.0f, 0.0f);
		glm::vec3 point(0.0f, -10.0f, -3.0f);
		glm::mat4 rtranslation1 = glm::mat4(1.0f);
		rtranslation1 = glm::translate(rtranslation1, -point);
		glm::mat4 rrotation = glm::mat4(1.0f);
		rrotation = glm::rotate(rrotation, glm::radians(rabbitAngle), axis);
		glm::mat4 rtranslation2 = glm::mat4(1.0f);
		rtranslation2 = glm::translate(rtranslation2, point);
		glm::mat4 rrotationMatrix = rtranslation2 * rrotation * rtranslation1;
		rabbitModelMatrix = rrotationMatrix * rabbitModelMatrix;

		rabbitModelMatrix = glm::translate(rabbitModelMatrix, glm::vec3(0.0f, -10.0f, -3.0f));
		drawModel("rabbit", shader.program, rabbitModelMatrix, viewMatrix, projectionMatrix);

		
		// TODO: Draw tortoise revolves around the clock tower (not (0, 0, 0))
		glm::mat4 tortoiseModelMatrix = glm::translate(groundModelMatrix, glm::vec3(18.0f, 1.5f, 0.0f));
		tortoiseModelMatrix = glm::scale(tortoiseModelMatrix, glm::vec3(0.2f, 0.2f, 0.2f));

		//tortoiseModelMatrix = glm::translate(tortoiseModelMatrix, glm::vec3(0.0f, 10.0f, 3.0f));

		tortoiseModelMatrix = glm::rotate(tortoiseModelMatrix, glm::radians(-180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		tortoiseModelMatrix = glm::rotate(tortoiseModelMatrix, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
		tortoiseModelMatrix = glm::rotate(tortoiseModelMatrix, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		
		//tortoiseModelMatrix = glm::rotate(tortoiseModelMatrix, glm::radians(tortoiseAngle), glm::vec3(0.0f, 1.0f, 0.0f));
		//tortoiseModelMatrix = glm::translate(tortoiseModelMatrix, glm::vec3(0.0f, -10.0f, -3.0f));


		glm::mat4 translation1 = glm::mat4(1.0f);
		translation1 = glm::translate(translation1, -point);
		glm::mat4 rotation = glm::mat4(1.0f);
		rotation = glm::rotate(rotation, glm::radians(tortoiseAngle), axis);
		glm::mat4 translation2 = glm::mat4(1.0f);
		translation2 = glm::translate(translation2, point);
		glm::mat4 rotationMatrix = translation2 * rotation * translation1;
		tortoiseModelMatrix = rotationMatrix * tortoiseModelMatrix;
		


		drawModel("tortoise", shader.program, tortoiseModelMatrix, viewMatrix, projectionMatrix);

		
		
		// TODO: Control speed and rotation
		
		if (isKey3Pressed) {
			rotateAngle += 0.5f;
		}



		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();
	return 0;
}

// TODO:
//	 1. press 1 to double the rotation speed
//   2. press 2 to halve the rotation speed
//   3. press 3 to rotate the clock tower




void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);

	if (key == GLFW_KEY_1 && action == GLFW_PRESS) {
		rotationTimes *= 2;
	}

	if (key == GLFW_KEY_2 && action == GLFW_PRESS) {
		rotationTimes /= 2;
	}

	if (key == GLFW_KEY_3 && action == GLFW_PRESS) {
		rotateAngle += 0.0f;
		if (!isKey3Pressed) {
			isKey3Pressed = true;
		}
		else {
			isKey3Pressed = false;
		}
	}
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
	windowWidth = width;
	windowHeight = height;
}

unsigned int modelVAO(Object& model)
{
	unsigned int VAO, VBO[3];
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glGenBuffers(3, VBO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT) * (model.positions.size()), &(model.positions[0]), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, 0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT) * (model.normals.size()), &(model.normals[0]), GL_STATIC_DRAW);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 3, 0);
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[2]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GL_FLOAT) * (model.texcoords.size()), &(model.texcoords[0]), GL_STATIC_DRAW);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(GL_FLOAT) * 2, 0);
	glEnableVertexAttribArray(2);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

	return VAO;
}

void drawModel(const string& target, unsigned int& shaderProgram, const glm::mat4& M, const glm::mat4& V, const glm::mat4& P)
{
	unsigned int mLoc, vLoc, pLoc;
	mLoc = glGetUniformLocation(shaderProgram, "M");
	vLoc = glGetUniformLocation(shaderProgram, "V");
	pLoc = glGetUniformLocation(shaderProgram, "P");
	glUniformMatrix4fv(mLoc, 1, GL_FALSE, glm::value_ptr(M));
	glUniformMatrix4fv(vLoc, 1, GL_FALSE, glm::value_ptr(V));
	glUniformMatrix4fv(pLoc, 1, GL_FALSE, glm::value_ptr(P));

	if (target == "rectangle")
	{
		glBindVertexArray(rectangleVAO);
		glDrawArrays(GL_TRIANGLES, 0, rectangleModel.positions.size());
	}
	else if (target == "triangle")
	{
		glBindVertexArray(triangleVAO);
		glDrawArrays(GL_TRIANGLES, 0, triangleModel.positions.size());
	}
	else if (target == "clock")
	{
		glBindVertexArray(clockVAO);
		glDrawArrays(GL_TRIANGLES, 0, clockModel.positions.size());
	}
	else if (target == "clock hand")
	{
		glBindVertexArray(clockHandVAO);
		glDrawArrays(GL_TRIANGLES, 0, clockHandModel.positions.size());
	}
	else if (target == "rabbit")
	{
		glBindVertexArray(rabbitVAO);
		glDrawArrays(GL_TRIANGLES, 0, rabbitModel.positions.size());
	}
	else if (target == "tortoise")
	{
		glBindVertexArray(tortoiseVAO);
		glDrawArrays(GL_TRIANGLES, 0, tortoiseModel.positions.size());
	}
	glBindVertexArray(0);
}