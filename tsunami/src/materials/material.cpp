#include "tsunami/materials/material.h"

GPUMaterial Material::pack() const {
	return m_gpu;
}