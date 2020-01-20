#pragma once
#include "stdafx.h"

std::string clErrInfo(cl::Error e);

struct Color
{
	uint8_t R, G, B, A;
};

Color white { 255, 255, 255, 255 };
Color black { 0, 0, 0, 255 };

void drawPNG(uint8_t *buffer, int res, string filename, Color yes = black, Color no = white);

void print2D(uint8_t *buffer, int res);