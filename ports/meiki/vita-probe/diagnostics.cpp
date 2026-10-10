// Record native allocation/exception failures before the SDK abort handler.
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <typeinfo>

extern "C" void *__real_malloc(size_t);
extern "C" void *__wrap_malloc(size_t bytes) {
    void *p = __real_malloc(bytes);
    if (!p && bytes) {
        struct mallinfo state = mallinfo();
        std::fprintf(
            stderr,
            "malloc failed: %u bytes, caller=%p; heap_used=%u heap_arena=%u\n",
            (unsigned)bytes, __builtin_return_address(0),
            (unsigned)state.uordblks, (unsigned)state.arena);
    }
    return p;
}
extern "C" [[noreturn]] void __real___cxa_throw(void *, void *,
                                                void (*)(void *));
extern "C" [[noreturn]] void __wrap___cxa_throw(void *object, void *type,
                                                void (*destroy)(void *)) {
    auto *info = static_cast<std::type_info *>(type);
    std::fprintf(stderr, "exception: %s, caller=%p\n",
                 info ? info->name() : "unknown", __builtin_return_address(0));
    __real___cxa_throw(object, type, destroy);
}
