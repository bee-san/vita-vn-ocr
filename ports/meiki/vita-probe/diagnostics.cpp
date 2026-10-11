// Record native allocation/exception failures before the SDK abort handler.
#include <cstdio>
#include <cstdlib>
#include <malloc.h>
#include <typeinfo>

extern "C" void *__real_malloc(size_t);
static int recording;
extern "C" void *__wrap_malloc(size_t bytes) {
    void *p = __real_malloc(bytes);
    if (!p && bytes && !__atomic_exchange_n(&recording, 1, __ATOMIC_ACQUIRE)) {
        struct mallinfo state = mallinfo();
        std::fprintf(
            stderr,
            "malloc failed: %u bytes, caller=%p; heap_used=%u heap_arena=%u\n",
            (unsigned)bytes, __builtin_return_address(0),
            (unsigned)state.uordblks, (unsigned)state.arena);
        __atomic_store_n(&recording, 0, __ATOMIC_RELEASE);
    }
    return p;
}
extern "C" [[noreturn]] void __real___cxa_throw(void *, void *,
                                                void (*)(void *));
extern "C" [[noreturn]] void __wrap___cxa_throw(void *object, void *type,
                                                void (*destroy)(void *)) {
    auto *info = static_cast<std::type_info *>(type);
    if (!__atomic_exchange_n(&recording, 1, __ATOMIC_ACQUIRE)) {
        std::fprintf(stderr, "exception: %s, caller=%p\n",
                     info ? info->name() : "unknown", __builtin_return_address(0));
        __atomic_store_n(&recording, 0, __ATOMIC_RELEASE);
    }
    __real___cxa_throw(object, type, destroy);
}
