// p 是用来更新内部 key 的明文nk0
// lsb、msb 分别表示取 dword 的最低和最高字节（不是位）
#include <cstdint>  // 提供固定大小的整數類型（例如 uint32_t）
#include <vector>   // 提供向量容器（例如 vector<uint8_t>）
#include <string>   // 提供字串類型（例如 string）
#include "crc.h"

const uint32_t MULINV = 0xd94fa8cd;

uint32_t k0 = 0x12345678;
uint32_t k1 = 0x23456789;
uint32_t k2 = 0x34567890;

void UpdateKeys(uint8_t p) {
    k0 = crc::Crc32(k0, p);
    k1 = (k1 + crc::lsb(k0)) * 134775813 + 1;
    k2 = crc::Crc32(k2, crc::msb(k1));
}

uint8_t k3() 
{
    uint16_t tmp = k2 | 3;
    return crc::lsb((tmp * (tmp ^ 1)) >> 8);
}

void InitialKeys(std::string pwd) 
{
    for (auto c: pwd)
        UpdateKeys(c);
}

void UpdateKeysBackward(uint8_t c)
{
    k2 = crc::Crc32Inv(k2, crc::msb(k1));
    k1 = (k1 - 1) * MULINV - crc::lsb(k0);
    uint32_t tmp = k2 | 3;
    k0 = crc::Crc32Inv(k0, c ^ crc::lsb(tmp * (tmp ^ 1) >> 8));
}


void Encrypt(std::vector<uint8_t>& data) 
{
    for (uint8_t& p : data) 
    {
        uint8_t c = p ^ k3();
        UpdateKeysBackward(c);
        p = c;
    }
}


void Decrypt(std::vector<uint8_t>& data) 
{
    for (uint8_t& c : data)
    {
        uint8_t p = c ^ k3();
        UpdateKeys(p);
        c = p;
    }
}

void InitTable() 
{
    for (int i = 0; i < 256; ++i) 
    {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) 
        {
            if (crc & 1)
                crc = crc >> 1 ^ 0xedb88320;  
            else
                crc = crc >> 1;
        }
        crc::CrcTable[i] = crc;
        crc::CrcInvTable[crc::msb(crc)] = crc << 8 ^ i;
    }
}

uint32_t Crc32(uint32_t pval, uint8_t b)
{
    return pval >> 8 ^ crc::CrcTable[crc::lsb(pval) ^ b];
}