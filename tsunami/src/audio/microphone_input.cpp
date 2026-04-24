// Purpose: Implements microphone capture, level extraction, and input device diagnostics.
#include <atomic>
#include <cmath>
#include <cstdint>

#include "miniaudio.h"

#include "tsunami/audio/microphone_input.h"

namespace {

constexpr ma_uint32 kCaptureChannelCount = 1;
constexpr ma_uint32 kSampleRate          = 48000;

}        // namespace

namespace audio {

struct MicrophoneInput::Impl {
	ma_device device{};

	std::atomic<float> latest_level{0.0f};
	bool               device_initialized = false;
	bool               available          = false;
	std::string        status_message     = "Microphone capture is unavailable.";
	std::string        device_name        = "Unavailable";

	static void dataCallback(ma_device* device, void* output, const void* input,
	                         ma_uint32 frame_count) {
		auto* impl = static_cast<Impl*>(device->pUserData);
		if (impl == nullptr) {
			return;
		}

		if (input == nullptr || frame_count == 0) {
			impl->latest_level.store(0.0f, std::memory_order_relaxed);
			(void) output;
			return;
		}

		const auto*     samples = static_cast<const float*>(input);
		const ma_uint32 channel_count =
		    device->capture.channels == 0 ? kCaptureChannelCount : device->capture.channels;
		const uint64_t sample_count =
		    static_cast<uint64_t>(frame_count) * static_cast<uint64_t>(channel_count);

		double sum_squares = 0.0;
		for (uint64_t sample_index = 0; sample_index < sample_count; ++sample_index) {
			const double sample = samples[sample_index];
			sum_squares += sample * sample;
		}

		const float rms =
		    static_cast<float>(std::sqrt(sum_squares / static_cast<double>(sample_count)));
		impl->latest_level.store(rms, std::memory_order_relaxed);

		(void) output;
	}
};

MicrophoneInput::MicrophoneInput() : m_impl(new Impl()) {
	ma_device_config config = ma_device_config_init(ma_device_type_capture);
	config.capture.format   = ma_format_f32;
	config.capture.channels = kCaptureChannelCount;
	config.sampleRate       = kSampleRate;
	config.dataCallback     = Impl::dataCallback;
	config.pUserData        = m_impl;

	const ma_result init_result = ma_device_init(nullptr, &config, &m_impl->device);
	if (init_result != MA_SUCCESS) {
		m_impl->status_message = "Failed to initialize the default microphone device.";
		return;
	}

	m_impl->device_initialized = true;

	char capture_name[MA_MAX_DEVICE_NAME_LENGTH + 1]{};
	if (ma_device_get_name(&m_impl->device, ma_device_type_capture, capture_name,
	                       sizeof(capture_name), nullptr) == MA_SUCCESS &&
	    capture_name[0] != '\0') {
		m_impl->device_name = capture_name;
	} else {
		m_impl->device_name = "Default microphone";
	}

	const ma_result start_result = ma_device_start(&m_impl->device);
	if (start_result != MA_SUCCESS) {
		m_impl->status_message = "Failed to start live microphone capture.";
		ma_device_uninit(&m_impl->device);
		m_impl->device_initialized = false;
		m_impl->device_name        = "Unavailable";
		return;
	}

	m_impl->available      = true;
	m_impl->status_message = "Live microphone capture active.";
}

MicrophoneInput::~MicrophoneInput() {
	if (m_impl != nullptr) {
		if (m_impl->device_initialized) {
			ma_device_uninit(&m_impl->device);
		}

		delete m_impl;
		m_impl = nullptr;
	}
}

bool MicrophoneInput::isAvailable() const {
	return m_impl != nullptr && m_impl->available;
}

float MicrophoneInput::latestLevel() const {
	return m_impl == nullptr ? 0.0f : m_impl->latest_level.load(std::memory_order_relaxed);
}

const std::string& MicrophoneInput::statusMessage() const {
	static const std::string kUnavailableMessage = "Microphone capture is unavailable.";
	return m_impl == nullptr ? kUnavailableMessage : m_impl->status_message;
}

const std::string& MicrophoneInput::deviceName() const {
	static const std::string kUnavailableDevice = "Unavailable";
	return m_impl == nullptr ? kUnavailableDevice : m_impl->device_name;
}

}        // namespace audio
