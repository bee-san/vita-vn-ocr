/* Host-built half of eval/textspot7_a9_driver.c: the C library's stderr for the Vita-compiled driver. */
#include <stdio.h>

void *t7_host_stderr(void);

void *t7_host_stderr(void)
{
    return stderr;
}
