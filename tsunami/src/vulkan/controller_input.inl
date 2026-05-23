//Poll the state of the controller
const ControllerInput::StateOfController ctrl = controller.poll();

//Debug print the names of connected joysticks
int joystick_count = 0;
for (int i = GLFW_JOYSTICK_1; i <= GLFW_JOYSTICK_16; ++i) {
    if (glfwJoystickPresent(i)) {
        std::cout << "[CTRL] slot " << i << ": " << glfwGetJoystickName(i)
                << " gamepad=" << glfwJoystickIsGamepad(i) << "\n";
        joystick_count++;
    }
}
std::cout << "[CTRL] total joysticks: " << joystick_count << "\n";

//Debug print the controller state to the console
if (ctrl.isConnected) {
    std::cout << "[CTRL] connected=" << ctrl.isConnected
        << " hasMapping=" << glfwJoystickIsGamepad(GLFW_JOYSTICK_1)
        << " lx=" << ctrl.left_stick_x
        << " ly=" << ctrl.left_stick_y
        << " L2=" << ctrl.left_trigger << "\n";
} else {
    std::cout << "[CTRL] not connected\n";
}

//Controller Additions Start Here for reading input and moving the camera
bool ctrl_moving = false;
if (ctrl.isConnected) {
    const bool has_stick_input =
        std::abs(ctrl.left_stick_x)  > 0.f || std::abs(ctrl.left_stick_y)  > 0.f ||
        std::abs(ctrl.right_stick_x) > 0.f || std::abs(ctrl.right_stick_y) > 0.f;

    //dPad left and dPad right adjust camera speed multiplier
    if (ctrl.dpad_left && !prev_dpad_left)
        fly_cam.m_speed = std::max(0.01f, fly_cam.m_speed * 0.5f);
    if (ctrl.dpad_right && !prev_dpad_right)
        fly_cam.m_speed *= 2.0f;
    prev_dpad_left  = ctrl.dpad_left;
    prev_dpad_right = ctrl.dpad_right;

    //Left and right shoulder buttons lower and raise the camera 
    if (ctrl.right_shoulder) {
        fly_cam.m_position.y += fly_cam.m_speed * dt;
        ctrl_moving = true;
    }
    if (ctrl.left_shoulder) {
        fly_cam.m_position.y -= fly_cam.m_speed * dt;
        ctrl_moving = true;
    }

    //PSbutton resets the camera to the initial position and orientation
    if (ctrl.touchpad_button && !prev_touchpad_button) {
        fly_cam.m_position = cam_reset_position;
        fly_cam.m_yaw      = cam_reset_yaw;
        fly_cam.m_pitch    = cam_reset_pitch;
        frame_number          = 0;
        needs_visibility_pass = true;
        reset_hipr_object_sampling();
    }
    prev_touchpad_button = ctrl.touchpad_button;

    //Camera Movement with the sticks
    if (has_stick_input) {
        fly_cam.applyControllerInput(ctrl.left_stick_x, ctrl.left_stick_y,
                                    ctrl.right_stick_x, ctrl.right_stick_y, dt);
        ctrl_moving = true;
    }

    //Both sticks clicked nukes the program and relaunches it. TODO: make sure it loads the same scene
    if (ctrl.left_stick_button && ctrl.right_stick_button) {
        char exe_path[MAX_PATH];
        GetModuleFileNameA(nullptr, exe_path, MAX_PATH);
        //launch with old scene path as argument
        ShellExecuteA(nullptr, "open", exe_path, m_scene_argument.empty() ? nullptr : m_scene_argument.c_str(), nullptr, SW_SHOWNORMAL);
        glfwSetWindowShouldClose(m_window->handle(), GLFW_TRUE);
    }

    //Toggle GUI with the options button
    if (ctrl.options_button && !prev_options_button) {
        show_all_gui = !show_all_gui;
        if (show_all_gui) {
            overlay_ctx.show_control_panel = true;
            show_selection_panel           = true;
        }
    }
    prev_options_button = ctrl.options_button;
}

const bool any_camera_moving = camera_moving_this_frame || ctrl_moving;


//When re-adding remember to change instances of camera_moving_this_frame to any_camera_moving and to set ctrl_moving to false at the end of each loop iteration
