/**
 *
 *  @file      raytrace.cpp
 *  @brief     Declaration of core raytracing functionality
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 */


 /*------------------Libraries---------------------*/
#include "renderer.h"
#include "kernels.cuh"
/*------------Varible initialzation---------------*/

/*------------Funcition Defenitions---------------*/

void Init(BYTE **pLpvBits, const RECT *window, HBITMAP *pHBitmap)
{
	int width = (*window).right;
	int height = (*window).bottom;

	BITMAPINFO bmi = {};
	bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bmi.bmiHeader.biWidth = width;
	bmi.bmiHeader.biHeight = -height; // Negative to have a top-down DIB
	bmi.bmiHeader.biPlanes = 1;
	bmi.bmiHeader.biBitCount = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	// Create the DIB section and obtain a pointer to the pixel buffer
	*pHBitmap = CreateDIBSection(NULL, &bmi, DIB_RGB_COLORS, (void **)&(*pLpvBits), NULL, 0);

	if (!(*pLpvBits) || !(*pHBitmap))
	{
		MessageBox(NULL, L"Could not allocate memory for bitmap", L"Error", MB_OK | MB_ICONERROR);
		exit(1);
	}

	// Initialize all pixels to black
	memset(*pLpvBits, 0, width * height * 4);


}


void Draw(BYTE **pLpvBits, const int &width, const int &height, Camera &cam)
{

	Draw_Caller(pLpvBits, &cam);

}

