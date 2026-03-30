#include "tsunami/scene/transform.h"

Transform::Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale) :
    m_position(position), m_rotation(rotation), m_scale(scale) {
	glm::mat4 T(1.0f);
	glm::mat4 R(1.0f);
	glm::mat4 S(1.0f);

	T = glm::translate(glm::mat4(1.0f), m_position);
	R = glm::rotate(R, glm::radians(rotation.x), glm::vec3(1, 0, 0));
	R = glm::rotate(R, glm::radians(rotation.y), glm::vec3(0, 1, 0));
	R = glm::rotate(R, glm::radians(rotation.z), glm::vec3(0, 0, 1));
	S = glm::scale(S, m_scale);

	m_transform = T * R * S;
}