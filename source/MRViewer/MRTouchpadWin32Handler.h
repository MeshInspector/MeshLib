#pragma once
#ifdef _WIN32

#include "MRTouchpadController.h"

#include <Windows.h>

#include <map>

namespace MR
{

class TouchpadWin32Handler : public TouchpadController::Handler
{
public:
	TouchpadWin32Handler( GLFWwindow* window );
	~TouchpadWin32Handler() override;

	static LRESULT WINAPI WindowSubclassProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam );

private:
	HWND window_;

	LONG_PTR glfwProc_;

	struct DeviceInfo
	{
		struct Cap
		{
			bool hasX{ false };
			bool hasY{ false };
			bool hasContactId{ false };
			bool hasTipSwitch{ false };
		};
		std::string deviceName;
		std::string preparsedData;
		std::map<USHORT, Cap> caps;
		USHORT contactCountLinkCollection{ 0 };
	};
	std::map<HANDLE, DeviceInfo> devices_;
	void fetchDeviceInfo_();

	std::map<ULONG, MR::Vector2ll> state_;

	static void processRawInput( TouchpadWin32Handler& handler, HRAWINPUT hRawInput );
};

} // namespace MR

#endif
