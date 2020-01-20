#pragma once
#include "stdafx.h"

std::string clErrInfo(cl::Error e);

struct Color
{
    uint8_t R, G, B, A;
};

constexpr Color white { 255, 255, 255, 255 };
constexpr Color black { 0, 0, 0, 255 };

void print2D(uint8_t *buffer, int res);
void drawPNG(uint8_t *buffer, int res, std::string filename, Color yes = black, Color no = white);