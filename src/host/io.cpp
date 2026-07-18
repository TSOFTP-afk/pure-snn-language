// =============================================================================
// io.cpp - 数据输入输出辅助
// =============================================================================

#include "network.h"
#include <fstream>
#include <iostream>

// 保存二进制数组
template<typename T>
void save_binary(const std::string& path, const T* data, size_t n) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return;
    }
    ofs.write(reinterpret_cast<const char*>(data), n * sizeof(T));
}

// 加载二进制数组
template<typename T>
void load_binary(const std::string& path, T* data, size_t n) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return;
    }
    ifs.read(reinterpret_cast<char*>(data), n * sizeof(T));
}

// 显式实例化（按需添加）
template void save_binary<float>(const std::string&, const float*, size_t);
template void load_binary<float>(const std::string&, float*, size_t);
