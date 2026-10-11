#include <stdio.h>
#include <libqhull_r/qhull_ra.h>

int main(void) {
    qhT state;
    qhT *qh = &state;
    coordT points[] = {0, 0, 1, 0, 1, 1, 0, 1};
    char flags[] = "qhull Qt";
    qh_zero(qh, stderr);
    int result = qh_new_qhull(qh, 2, 4, points, 0, flags, NULL, stderr);
    if (result || qh->num_facets != 4) return 1;
    qh_freeqhull(qh, !qh_ALL);
    int current, total;
    qh_memfreeshort(qh, &current, &total);
    if (current || total) return 2;
    puts("qhull: square hull passed");
    return 0;
}
