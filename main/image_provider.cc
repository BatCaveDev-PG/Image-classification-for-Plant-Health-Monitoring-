
/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_system.h"
#include "esp_timer.h"

#include "app_camera_esp.h"
#include "esp_camera.h"
#include "model_settings.h"
#include "image_provider.h"
#include "esp_main.h"

static const char* TAG = "app_camera";

static uint16_t *display_buf; // buffer to hold data to be sent to display

// Get the camera module ready
TfLiteStatus InitCamera() {
#if CLI_ONLY_INFERENCE
  ESP_LOGI(TAG, "CLI_ONLY_INFERENCE enabled, skipping camera init");
  return kTfLiteOk;
#endif
// if display support is present, initialise display buf
#if DISPLAY_SUPPORT
  if (display_buf == NULL) {
    // Size of display_buf:
    // Frame 96x96 from camera is extrapolated to 192x192. RGB565 pixel format -> 2 bytes per pixel
    display_buf = (uint16_t *) heap_caps_malloc(96 * 2 * 96 * 2 * sizeof(uint16_t), MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  }
  if (display_buf == NULL) {
    ESP_LOGE(TAG, "Couldn't allocate display buffer");
    return kTfLiteError;
  }
#endif // DISPLAY_SUPPORT

#if ESP_CAMERA_SUPPORTED
  int ret = app_camera_init();
  if (ret != 0) {
    MicroPrintf("Camera init failed\n");
    return kTfLiteError;
  }
  MicroPrintf("Camera Initialized\n");
#else
  ESP_LOGE(TAG, "Camera not supported for this device");
#endif
  return kTfLiteOk;
}

void *image_provider_get_display_buf()
{
  return (void *) display_buf;
}

// Get an image from the camera module
TfLiteStatus GetImage(int image_width, int image_height, int channels, int8_t* image_data) {
#if ESP_CAMERA_SUPPORTED
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    ESP_LOGE(TAG, "Camera capture failed");
    return kTfLiteError;
  }

#if DISPLAY_SUPPORT
  // In case if display support is enabled, we initialise camera in rgb mode
  // For display we extra-polate the data to 192X192
  for (int i = 0; i < kNumRows; i++) {
    for (int j = 0; j < kNumCols; j++) {
      uint16_t pixel = ((uint16_t *) (fb->buf))[i * kNumCols + j];

      // Extract RGB components from the RGB565 pixel format for inference
      uint8_t hb = pixel & 0xFF;
      uint8_t lb = pixel >> 8;
      uint8_t r = (lb & 0x1F) << 3;
      uint8_t g = ((hb & 0x07) << 5) | ((lb & 0xE0) >> 3);
      uint8_t b = (hb & 0xF8);

      // Quantize the pixel values to int8 range [-128, 127] and store in image_data
      image_data[(i * kNumCols + j) * channels] = (r - 128);  // Red channel
      image_data[(i * kNumCols + j) * channels + 1] = (g - 128);  // Green channel
      image_data[(i * kNumCols + j) * channels + 2] = (b - 128);  // Blue channel


      // to display
      display_buf[2 * i * kNumCols * 2 + 2 * j] = pixel;
      display_buf[2 * i * kNumCols * 2 + 2 * j + 1] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j] = pixel;
      display_buf[(2 * i + 1) * kNumCols * 2 + 2 * j + 1] = pixel;
    }
  }
#else // DISPLAY_SUPPORT
  MicroPrintf("Image Captured\n");
  // If the camera is initialized to RGB, we resize and normalize the image for inference
  for (int i = 0; i < kNumCols * kNumRows * channels; i++) {
    image_data[i] = ((uint8_t *) fb->buf)[i] - 128;  // Quantize to int8 range
  }
#endif // DISPLAY_SUPPORT

  esp_camera_fb_return(fb);
  return kTfLiteOk;
#else
  return kTfLiteError;
#endif
}