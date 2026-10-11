#include "model_reader.h"

int meiki_model_reader_valid(const MeikiModelReader *reader)
{
    /* Check before field access: pointers inside the descriptor require natural
     * alignment on ARM. sizeof-based alignment guesses fail on 64-bit hosts. */
    return reader && !((uintptr_t)reader % __alignof__(MeikiModelReader)) &&
           reader->size == sizeof(*reader) && reader->read &&
           reader->byte_count == MEIKI_MODEL_BYTES;
}

int meiki_model_read(const MeikiModelReader *reader, void *destination,
                     size_t capacity, MeikiStats *stats)
{
    if (!meiki_model_reader_valid(reader) || !destination || !stats ||
        capacity < reader->byte_count ||
        (uintptr_t)destination > UINTPTR_MAX - reader->byte_count)
        return -1;
    uint32_t offset = 0;
    while (offset < reader->byte_count) {
        uint32_t bytes = reader->byte_count - offset;
        if (bytes > MEIKI_MODEL_READ_CHUNK) bytes = MEIKI_MODEL_READ_CHUNK;
        int rc = reader->read(reader->context, offset,
                              (unsigned char *)destination + offset, bytes);
        if (rc) {
            stats->model_read_error = rc;
            return -14;
        }
        offset += bytes;
        stats->model_read_bytes = offset;
    }
    return 0;
}
