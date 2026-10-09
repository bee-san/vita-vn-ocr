// vocr_ncnn_rec: run a PaddleOCR CTC recogniser in ncnn exactly as vita-jp-overlay's
// core/local_ocr.cpp does, for eval/run.py's ncnn:DIR backend.
//
//   vocr_ncnn_rec MODEL.param MODEL.bin < crops.txt
//
// stdin: one binary PPM (P6, RGB) path per line. stdout: one line per crop with the
// greedy-CTC token ids (blank 0 dropped, repeats collapsed), space separated; an empty
// line when nothing was read. Preprocessing: RGBA -> BGR, resize to height 48 and
// width (w*48 + h/2)/h clamped to [8, 1024] with ncnn's from_pixels_resize, then
// (p - 127.5) / 127.5. Options: one thread, lightmode, fp16/bf16 off, winograd and
// sgemm off. stderr ends with "lines=N maxrss_kib=K".
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/resource.h>

#include "net.h"

static bool read_ppm(const char *path, std::vector<unsigned char> &rgba, int &w, int &h)
{
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    int maxv = 0;
    char magic[3] = {0};
    bool ok = fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxv) == 4 && !strcmp(magic, "P6") &&
              maxv == 255 && w > 0 && h > 0 && fgetc(f) != EOF;
    std::vector<unsigned char> rgb;
    if (ok) {
        rgb.resize((size_t)w * h * 3);
        ok = fread(rgb.data(), 1, rgb.size(), f) == rgb.size();
    }
    fclose(f);
    if (!ok) return false;
    rgba.resize((size_t)w * h * 4);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgba[4 * i] = rgb[3 * i];
        rgba[4 * i + 1] = rgb[3 * i + 1];
        rgba[4 * i + 2] = rgb[3 * i + 2];
        rgba[4 * i + 3] = 255;
    }
    return true;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s MODEL.param MODEL.bin < crop_paths\n", argv[0]);
        return 2;
    }
    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.num_threads = 1;
    net.opt.lightmode = true;
    net.opt.use_fp16_arithmetic = net.opt.use_fp16_storage = net.opt.use_fp16_packed = false;
    net.opt.use_bf16_storage = false;
    net.opt.use_winograd_convolution = false;
    net.opt.use_sgemm_convolution = false;
    if (net.load_param(argv[1]) || net.load_model(argv[2])) {
        fprintf(stderr, "load failed\n");
        return 3;
    }
    char buf[4096];
    int n = 0;
    while (fgets(buf, sizeof buf, stdin)) {
        size_t len = strlen(buf);
        while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = 0;
        if (!len) continue;
        std::vector<unsigned char> rgba;
        int w = 0, h = 0;
        if (!read_ppm(buf, rgba, w, h)) {
            fprintf(stderr, "bad ppm %s\n", buf);
            return 4;
        }
        int width = (w * 48 + h / 2) / h;
        if (width < 8) width = 8;
        if (width > 1024) width = 1024;
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgba.data(), ncnn::Mat::PIXEL_RGBA2BGR, w, h, width, 48);
        for (int c = 0; c < 3; c++) {
            float *p = in.channel(c);
            for (int i = 0; i < in.w * in.h; i++) p[i] = (p[i] - 127.5f) / 127.5f;
        }
        ncnn::Extractor ex = net.create_extractor();
        ncnn::Mat out;
        if (ex.input("in0", in) || ex.extract("out0", out) || out.dims != 2) {
            fprintf(stderr, "extract failed on %s\n", buf);
            return 5;
        }
        std::string line;
        int last = 0;
        for (int y = 0; y < out.h; y++) {
            const float *s = out.row(y);
            int t = 0;
            for (int i = 1; i < out.w; i++)
                if (s[i] > s[t]) t = i;
            if (t && t != last) {
                if (!line.empty()) line += ' ';
                line += std::to_string(t);
            }
            last = t;
        }
        printf("%s\n", line.c_str());
        fflush(stdout);
        n++;
    }
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    fprintf(stderr, "lines=%d maxrss_kib=%ld\n", n, ru.ru_maxrss);
    return 0;
}
