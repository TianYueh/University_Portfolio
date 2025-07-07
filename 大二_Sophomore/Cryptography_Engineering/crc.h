#ifndef CRC_H
#define CRC_H

#include<cstdint>
namespace crc {
    extern uint32_t CrcTable[256];

    // CRC32 反向表格資料
    extern uint32_t CrcInvTable[256] ;

    constexpr uint8_t msb(uint32_t value)
    {
        return static_cast<uint8_t>((value >> 24) & 0xFF);
    }

    constexpr uint8_t lsb(uint32_t value)
    {
        return static_cast<uint8_t>(value & 0xFF);
    }

    uint32_t Crc32(uint32_t pval, uint8_t b);

    uint32_t Crc32Inv(uint32_t pval, uint8_t b);
}
#endif  