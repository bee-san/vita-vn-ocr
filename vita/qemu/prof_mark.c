/* prof_mark.c - the phase marker of the per-layer cycle profile (vita/a9_layers.py), linked into
 * the qemu build of vocr_cli only, never into a console build. runtime/vocr.c built with
 * -DVOCR_PROF=vocr_prof_mark calls it as each phase of vocr_run() starts; vita/qemu/insn_mix.c
 * (marker=<this function's address>) reads the phase id from r0 on entry. It lives in its own
 * translation unit so that the call cannot be inlined away. */
void vocr_prof_mark(unsigned id);

void vocr_prof_mark(unsigned id)
{
    __asm__ volatile("" : : "r"(id) : "memory");
}
