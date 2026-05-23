#include "ControllerInput.h"
#include "../../../vendors/glfw/include/GLFW/glfw3.h"
#include <cmath>
#include <iostream>

namespace ControllerInput {

Controller::Controller(int joystick_id) : m_joystick_id(joystick_id) {}

float Controller::applyDeadzone(float value, float dz) {
    if (std::abs(value) < dz) return 0.0f;
    const float sign = value > 0.0f ? 1.0f : -1.0f;
    return sign * (std::abs(value) - dz) / (1.0f - dz);
}

float Controller::applySensitivity(float value, float sensitivity) {
    if (value == 0.0f) return 0.0f;
    const float sign = value > 0.0f ? 1.0f : -1.0f;
    return sign * std::pow(std::abs(value), sensitivity);
}

float Controller::normalizeTrigger(float value, float threshold) {
    const float normalized = (value + 1.0f) * 0.5f;  // [0, 1]
    return normalized > threshold ? (normalized - threshold) / (1.0f - threshold) : 0.0f;
}

void Controller::getControllerMappingsFromDB(const std::string& mappings_db_text) {
    if (mappings_db_text.empty()) {
        std::cout << "[CONTROLLER] No mappings string provided — using GLFW built-ins only\n";
        return;
    }
    const int result = glfwUpdateGamepadMappings(mappings_db_text.c_str());
    if (result == GLFW_TRUE) {
        std::cout << "[CONTROLLER] Gamepad mappings loaded successfully\n";
    } else {
        std::cerr << "[CONTROLLER] glfwUpdateGamepadMappings failed\n";
    }
}


//Reads the state of the controller and returns it as a struct
StateOfController Controller::poll() {
    StateOfController state{};

    state.isConnected = glfwJoystickPresent(m_joystick_id) == GLFW_TRUE;
    if (!state.isConnected) {
        previous_state = state;
        return state;
    }

    const char* name = glfwGetJoystickName(m_joystick_id);
    state.name = name ? name : "Unknown";

    GLFWgamepadstate gp{};
    glfwGetGamepadState(m_joystick_id, &gp);

    // Sticks — apply deadzone then sensitivity
    auto processStick = [&](float raw) {
        return applySensitivity(applyDeadzone(raw, deadzone_threshold), stick_sensitivity);
    };

    state.left_stick_x  = processStick(gp.axes[GLFW_GAMEPAD_AXIS_LEFT_X]);
    state.left_stick_y  = processStick(-gp.axes[GLFW_GAMEPAD_AXIS_LEFT_Y]);  // flip: up = positive
    state.right_stick_x = processStick(gp.axes[GLFW_GAMEPAD_AXIS_RIGHT_X]);
    state.right_stick_y = processStick(gp.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]);

    // Triggers
    state.left_trigger  = normalizeTrigger(gp.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER],  trigger_threshold);
    state.right_trigger = normalizeTrigger(gp.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER], trigger_threshold);

    // D-pad
    state.dpad_up    = gp.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP]    == GLFW_PRESS;
    state.dpad_down  = gp.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN]  == GLFW_PRESS;
    state.dpad_left  = gp.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT]  == GLFW_PRESS;
    state.dpad_right = gp.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] == GLFW_PRESS;

    // Face buttons
    state.south_button = gp.buttons[GLFW_GAMEPAD_BUTTON_CROSS]    == GLFW_PRESS;
    state.east_button  = gp.buttons[GLFW_GAMEPAD_BUTTON_CIRCLE]   == GLFW_PRESS;
    state.west_button  = gp.buttons[GLFW_GAMEPAD_BUTTON_SQUARE]   == GLFW_PRESS;
    state.north_button = gp.buttons[GLFW_GAMEPAD_BUTTON_TRIANGLE] == GLFW_PRESS;

    // Shoulders
    state.left_shoulder  = gp.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER]  == GLFW_PRESS;
    state.right_shoulder = gp.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] == GLFW_PRESS;

    // Stick clicks
    state.left_stick_button  = gp.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB]  == GLFW_PRESS;
    state.right_stick_button = gp.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_THUMB] == GLFW_PRESS;

    // Menu
    state.options_button  = gp.buttons[GLFW_GAMEPAD_BUTTON_START] == GLFW_PRESS;
    state.touchpad_button = gp.buttons[GLFW_GAMEPAD_BUTTON_GUIDE] == GLFW_PRESS; //Is actually the playstation button, apparently GLFW doesn't have the touchpad mapped.

    previous_state = state;
    return state;
}

} // namespace ControllerInput