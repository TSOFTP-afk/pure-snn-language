// main.cpp - stage 1 entry point
#include <cstdio>

// defined in bptt_demo.cu
void run_bptt_demo();

int main() {
    printf("============================================================\n");
    printf("  SNN Stage 1: BPTT + Surrogate Gradient - Grad Check + Learning Demo\n");
    printf("============================================================\n\n");
    run_bptt_demo();
    return 0;
}
