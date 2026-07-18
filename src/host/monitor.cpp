// =============================================================================
// monitor.cpp - 监控和日志
// =============================================================================

#include "network.h"
#include <iostream>
#include <fstream>
#include <ctime>

void log_stats(const NetworkStats& stats, int episode, int step) {
    static std::ofstream log_file;
    static bool initialized = false;

    if (!initialized) {
        log_file.open("training_log.csv");
        if (log_file.is_open()) {
            log_file << "episode,step,total_spikes,exc_spikes,inh_spikes,"
                     << "mean_fire_rate,mean_weight,dopamine\n";
            initialized = true;
        }
    }

    if (log_file.is_open()) {
        log_file << episode << "," << step << ","
                 << stats.total_spikes << ","
                 << stats.excitatory_spikes << ","
                 << stats.inhibitory_spikes << ","
                 << stats.mean_fire_rate << ","
                 << stats.mean_weight << ","
                 << stats.dopamine_level << "\n";
        log_file.flush();
    }
}

void print_timestamp(const std::string& label) {
    time_t now = time(nullptr);
    char buf[64];
    struct tm tm_info;
    localtime_s(&tm_info, &now);
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_info);
    std::cout << "[" << buf << "] " << label << std::endl;
}
