#include "tsunami/scene/transform.h"

class Camera {
  public:
	Camera();
	~Camera() = default;

	void      setTransform(const Transform& transform);
	Transform getTransform() const;

  private:
	Transform m_transform;
	glm::vec3 m_target;
	glm::vec3 m_position;
	glm::vec3 m_up;
	float     m_fov;
	float     m_nearClip;
	float     m_farClip;
};