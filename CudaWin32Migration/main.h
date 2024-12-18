/*******************************************************************************
 *
 *  @file		main.h
 *  @brief		Main program entry point
 *  @author		Johan Karlsson - github.com/dosalock
 *  @date		8.11.2024
 *  @copyright	Copyright © [2024] [Johan Karlsson]
 *
 ******************************************************************************/
#pragma once


 /*-------------------Defines----------------------*/
#define _USE_MATH_DEFINES

/*------------------Libraries---------------------*/

#include <Windows.h>
#include "renderer.h"

/*----------------My Libraries---------------------*/



/*-------------Function Declarations-------------*/


int WINAPI WinMain(_In_ HINSTANCE hInstance,
				   _In_opt_ HINSTANCE hPrevInstance,
				   _In_ LPSTR lpCmdLine,
				   _In_ int nCmdShow);


LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam);
/*---------------------------------------*/
/*---------------------------------------*/
/*---------------------------------------*/


