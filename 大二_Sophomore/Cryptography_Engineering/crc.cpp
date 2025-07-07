#include "crc.h"

namespace crc {

    // CRC32 表格資料
     uint32_t CrcTable[256] = {
        // 這裡放置 CRC32 表格資料
    };

    // CRC32 反向表格資料
    uint32_t CrcInvTable[256] = {
        // 這裡放置 CRC32 反向表格資料
    };

    uint32_t Crc32(uint32_t pval, uint8_t b) {
        return pval >> 8 ^ CrcTable[lsb(pval) ^ b];
    }

    uint32_t Crc32Inv(uint32_t pval, uint8_t b) {
        return pval << 8 ^ CrcInvTable[msb(pval) >> 24] ^ b;
    }

} // namespace crc