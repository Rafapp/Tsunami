#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Transform {
  public:
	Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale);
	~Transform() = default;

	glm::mat4 getTransform();
	void updateTransform();

	void setPosition(const glm::vec3& position) { m_position = position; updateTransform();}
	void setRotation(const glm::vec3& rotation) { m_rotation = rotation; updateTransform();}
	void setScale(const glm::vec3& scale) { m_scale = scale; updateTransform();}

	glm::mat4 pack() const { return m_transform; }

	glm::vec3 m_position;
	glm::vec3 m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_transform;
};