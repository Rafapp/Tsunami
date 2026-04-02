#include "tsunami/scene/transform.h"

Transform::Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale) {
	m_position = position;
	m_rotation = rotation;
	m_scale    = scale;
	updateTransform();
}

void Transform::updateTransform() {
	glm::mat4 T        = glm::translate(glm::mat4(1.0f), m_position);
	glm::mat4 R        = glm::toMat4(glm::quat(glm::radians(m_rotation)));
	glm::mat4 S        = glm::scale(glm::mat4(1.0f), m_scale);
	m_transform        = glm::transpose(T * R * S);
	m_inverseTransform = glm::transpose(glm::inverse(m_transform));
}

glm::mat4 Transform::getTransform() {
	updateTransform();
	m_transform        = glm::transpose(m_transform);
	m_inverseTransform = glm::transpose(m_inverseTransform);
	return m_transform;
}