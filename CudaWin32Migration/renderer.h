/*******************************************************************************
 *
 *  @file      raytrace.h
 *  @brief     Raytrace functions, draw loop vector calculations
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      8.11.2024
 *  @copyright Copyright © [2024] [Johan Karlsson]
 *
 ******************************************************************************/
#pragma once

 /*------------------Includes---------------------*/
#include "object_structs.h"
#include <Windows.h>

/*------------Variable Declarations---------------*/

/*------------Function Declarations---------------*/
/**
 * @brief Main draw function, sets all the pixel values
 * @param[in,out] pLpvBits - Pointer to buffer of viewport pixels
 * @param[in] width - Viewport width in pixels
 * @param[in] height - Viewport height in pixels
 */
void Draw(BYTE **pLpvBits, const int &width, const int &height, Camera &cam);

/**
 * @brief Initzialises the scene, bitmap height & width etc.
 * @param[in,out] pLpvBits - Pointer to buffer of viewport pixels
 * @param[in] window - Handle to rectangle of viewport
 * @param[in,out] pHBitmap - Handle to a bitmap
 */
void Init(BYTE **pLpvBits, const RECT *window, HBITMAP *pHBitmap);


