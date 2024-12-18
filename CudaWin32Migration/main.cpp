/**
 *
 *  @file      main.cpp
 *  @brief	   Application entry
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 */


/*------------------Includes---------------------*/
#include "main.h"
#include "cuda_manager.h"
#include <iostream>
/*------------Varible initialzation--------------*/

HBITMAP hBitmap = NULL;
HDC hdcWindow   = NULL;
BYTE *lpvBits   = NULL;
RECT window     = { };
Camera cam      = { };
int height      = 0;
int width       = 0;
// std::chrono::time_point deltaTime = std::chrono::steady_clock().now();
bool cameraIsMoving = false;
bool button_pressed = false;

/*-------------Function Definitions--------------*/

/**
 * @brief Application entrypoint.
 *
 * @param hInstance Handle to instance, base address of module memory
 * @param hPrevInstance Handle to previous instance - always NULL
 *		  if you need to detect if another exists, use CreateMutex
 *		  returns ERROR_ALREADY_EXISTS if theres already one named the same
 * @param lpCmdLine String to the command line for the application
 * @param nCmdShow Controls how the window is to be shown
 * @return
 */
int WINAPI WinMain ( _In_ HINSTANCE hInstance,
					 _In_opt_ HINSTANCE hPrevInstance,
					 _In_ LPSTR lpCmdLine,
					 _In_ int nCmdShow )
{
	// Register window class
	const wchar_t CLASS_NAME[] = L"Window";


	WNDCLASS WindowClass = { };

	WindowClass.lpfnWndProc   = WindowProc;
	WindowClass.hInstance     = hInstance;
	WindowClass.lpszClassName = CLASS_NAME;


	RegisterClass( &WindowClass );

	// Create window
	HWND hwnd = CreateWindowEx( 0,
								CLASS_NAME,
								L"Win32 Raytracer",
								WS_OVERLAPPEDWINDOW, // Style of window

								CW_USEDEFAULT,
								CW_USEDEFAULT,
								1040,
								1063,

								NULL,
								NULL,
								hInstance,
								NULL );

	if ( hwnd == NULL )
	{
		return 0;
	}

	GetClientRect( hwnd, &window );
	Init( &lpvBits, &window, &hBitmap );
	ShowWindow( hwnd, nCmdShow );
	UpdateWindow( hwnd );

	MSG msg = { };

	while ( GetMessage( &msg, NULL, 0, 0 ) > 0 )
	{
		TranslateMessage( &msg );
		DispatchMessage( &msg );
	}

	return 0;
}

/**
 * Called from message dispatch, determines response to messages.
 *
 * @param hwnd handle to the window
 * @param uMsg message code; e.g. WM_KEYDOWN
 * @param wParam data pertaining to message e.g. which key pressen on WM_KEYDOWN
 * @param lParam data pertaining to message if neeeded
 * @return
 */
LRESULT CALLBACK WindowProc ( HWND hwnd,
							  UINT uMsg,
							  WPARAM wParam,
							  LPARAM lParam )
{
	double secondsPerFrame = 3'300'000;
	float moveSpeed        = 0.1f;
	float rotationSpeed    = 2.0f;
	width                  = window.right;
	height                 = window.bottom;

	switch ( uMsg )
	{
		case WM_CREATE:
		{
			break;
		}
		case WM_DESTROY:
		{
			// Cleanup
			if ( hBitmap )
			{
				DeleteObject( hBitmap );
			}
			PostQuitMessage( 0 );

			break;
		}

		case WM_PAINT:
		{
			PAINTSTRUCT ps;

			GetClientRect( hwnd, &window ); // Use client area dimensions
			width   = window.right;
			height  = window.bottom;
			HDC hdc = BeginPaint( hwnd, &ps );

			HDC hdcMem        = CreateCompatibleDC( hdc );
			HGDIOBJ oldBitmap = SelectObject( hdcMem, hBitmap );

			BitBlt( hdc, 0, 0, width, height, hdcMem, 0, 0, SRCCOPY );

			SelectObject( hdcMem, oldBitmap );
			DeleteDC( hdcMem );

			// All painting occurs here, between BeginPaint and EndPaint.

			EndPaint( hwnd, &ps );

			break;
		}
		case WM_MOVE:
		{
			// TODO: do we want to do something on move?
		}
		case WM_KEYUP:
		{
			cameraIsMoving = false;
			break;
		}
		case WM_KEYDOWN:
		{
			switch ( wParam )
			{
				case 'W':
				{
					// cam.MoveForward(moveSpeed);
					break;
				}
				case 'A':
				{
					// cam.MoveSideways(moveSpeed);
					break;
				}
				case 'S':
				{
					// cam.MoveForward(-moveSpeed);
					break;
				}
				case 'D':
				{
					// cam.MoveSideways(-moveSpeed);
					break;
				}
				case 'Q':
				{
					// cam.yaw -= rotationSpeed;
					break;
				}
				case 'E':
				{
					// cam.yaw += rotationSpeed;
					break;
				}
				case 'M':
				{
					try
					{
						CUDAManager::GetInstance( ).Initialize( );


						auto devices =
						  CUDAManager::GetInstance( ).GetAvailableDevices( );
						for ( const auto &device : devices )
						{
							std::cout << "Device:" << device.name
									  << ", Compute Capability: "
									  << device.computeCapability << std::endl;
						}
						BYTE *twopv_bits =
						  ( BYTE *)malloc( width*height*4);
						CUDAManager::GetInstance( ).ProcessGPUData(
						  lpvBits,
						  twopv_bits,
						  width * height * 4 );
						memcpy( lpvBits, twopv_bits, width*height*4);
						InvalidateRect( hwnd, NULL, TRUE );
						free( twopv_bits);
					}
					catch ( const std::exception &e )
					{
						std::cerr << "CUDA Error: " << e.what( ) << std::endl;
					}


					
					button_pressed = true;
					break;
				}
				default:
					break;
			}
			break;
		}
		default:
		{
			return DefWindowProc( hwnd, uMsg, wParam, lParam );
		}
	}
	return DefWindowProc( hwnd, uMsg, wParam, lParam );
}
