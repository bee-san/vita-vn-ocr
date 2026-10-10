// Host probe for the existing Meiki recognizer, not an OCR implementation.
// Input on stdin: paths to float32 NCHW tensors prepared with Meiki's code.
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::unique_ptr<MNN::Interpreter> net(MNN::Interpreter::createFromFile(argv[1]));
    if (!net) return 3;
    MNN::BackendConfig backend;
    backend.memory = MNN::BackendConfig::Memory_Low;
    backend.precision = MNN::BackendConfig::Precision_Normal;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    config.backendConfig = &backend;
    auto *session = net->createSession(config);
    if (!session) return 4;
    auto *input = net->getSessionInput(session, "images");
    if (!input || input->elementSize() != 3 * 32 * 960) return 5;
    MNN::Tensor host_input(input, MNN::Tensor::CAFFE);
    auto *codes = net->getSessionOutput(session, "char_codes");
    auto *boxes = net->getSessionOutput(session, "boxes");
    auto *scores = net->getSessionOutput(session, "scores");
    if (!codes || !boxes || !scores) return 6;
    net->releaseModel();
    float memory = 0;
    net->getSessionInfo(session, MNN::Interpreter::MEMORY, &memory);
    fprintf(stderr, "session_memory_mib=%.6f (excludes model and process overhead)\n", memory);
    char path[4096];
    while (fgets(path, sizeof(path), stdin)) {
        std::string filename(path);
        while (!filename.empty() && (filename.back() == '\n' || filename.back() == '\r')) filename.pop_back();
        FILE *file = fopen(filename.c_str(), "rb");
        if (!file) return 7;
        size_t bytes = fread(host_input.host<float>(), 1, host_input.size(), file);
        bool exact = bytes == (size_t)host_input.size() && fgetc(file) == EOF;
        fclose(file);
        if (!exact || !input->copyFromHostTensor(&host_input)) return 8;
        auto started = std::chrono::steady_clock::now();
        auto rc = net->runSession(session);
        if (rc != MNN::NO_ERROR) return 9;
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        MNN::Tensor host_codes(codes, codes->getDimensionType());
        MNN::Tensor host_boxes(boxes, boxes->getDimensionType());
        MNN::Tensor host_scores(scores, scores->getDimensionType());
        if (!codes->copyToHostTensor(&host_codes) || !boxes->copyToHostTensor(&host_boxes) || !scores->copyToHostTensor(&host_scores)) return 10;
        if (host_codes.elementSize() != 48 || host_boxes.elementSize() != 192 || host_scores.elementSize() != 48 || host_codes.getType().bits != 32) return 11;
        printf("{\"ms\":%.6f,\"codes\":[", ms);
        for (int i = 0; i < 48; ++i) printf("%s%d", i ? "," : "", host_codes.host<int32_t>()[i]);
        printf("],\"boxes\":[");
        for (int i = 0; i < 192; ++i) printf("%s%.9g", i ? "," : "", host_boxes.host<float>()[i]);
        printf("],\"scores\":[");
        for (int i = 0; i < 48; ++i) printf("%s%.9g", i ? "," : "", host_scores.host<float>()[i]);
        printf("]}\n");
        fflush(stdout);
    }
    return ferror(stdin) ? 12 : 0;
}
