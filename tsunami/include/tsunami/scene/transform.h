#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

class Transform {
  public:
	Transform(glm::vec3 position, glm::vec3 rotation, glm::vec3 scale);
	virtual ~Transform() = default;
	glm::mat4 getTransform() {
		return m_transform;
	};

  private:
	glm::vec3 m_position;
	glm::vec3 m_rotation;
	glm::vec3 m_scale;
	glm::mat4 m_transform;
};