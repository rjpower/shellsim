extern int main_value;
extern int main_callback(int);
int shared_value = 7;
int (*stored_callback)(int) = main_callback;
int library_add(int n) { shared_value += n; return stored_callback(shared_value) + main_value; }
__attribute__((constructor)) static void initialize(void) { shared_value += 3; }
