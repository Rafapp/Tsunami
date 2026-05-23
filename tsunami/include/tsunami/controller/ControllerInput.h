#pragma once
#include "../../include/tsunami/controller/ControllerInput.h"
#include <cstdint>
#include <string>


//The TLDR of this is that we create a controller struct by polling GLFW for the state of the controller and pack it into a struct
//The inputs also get some quality of life improvements such as deadzone and sensitivity adjustments, and trigger normalization
//Then in the main loop all we have to do is declare a controller, poll it, and read values from the struct to move the camera around

//TODO: I want to make the controller movmenent a function in here, so the main loop just calls a readController function or something
//TODO: Also figure out the weird header issue I am having

namespace ControllerInput{

    struct StateOfController{

        //Read info
        bool isConnected;
        std::string name;
        int32_t buttonCount;
        int32_t axisCount;

        //Sticks
        float left_stick_x = 0.0f;
        float left_stick_y = 0.0f;
        float right_stick_x = 0.0f;
        float right_stick_y = 0.0f;

        //Dpad
        bool dpad_up = false;
        bool dpad_down = false;
        bool dpad_left = false;
        bool dpad_right = false;

        //Face Buttons
        bool north_button = false; //Triangle (in Playstation terms)
        bool south_button = false; //Cross
        bool west_button = false;  //Square
        bool east_button = false;  //Circle

        //Shoulder Buttons
        bool left_shoulder = false;
        bool right_shoulder = false;

        //Trigger Buttons
        float left_trigger = 0.0f;
        float right_trigger = 0.0f;

        //Thumbstick Buttons
        bool left_stick_button = false;
        bool right_stick_button = false;

        //Menu Buttons
        bool options_button = false;
        bool touchpad_button = false;
    };


    class Controller{

    public:
        //functions
        explicit Controller(int joystick_id = 0);
        static void getControllerMappingsFromDB(const std::string& db_path);
        StateOfController poll();
        int getStickID() const { return m_joystick_id; }
        //public data
        float deadzone_threshold = 0.1f;
        float trigger_threshold = 0.5f;
        float stick_sensitivity = 1.0f;

    private:
        int m_joystick_id;
        StateOfController previous_state{};
        static float applyDeadzone(float value, float threshold);
        static float applySensitivity(float value, float sensitivity);
        static float normalizeTrigger(float value, float threshold);
    };

}// end of namespace ControllerInput