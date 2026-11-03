//! 极简只读 FAT16/FAT32 驱动（8.3 短文件名）。
//!
//! 为什么不用 `fatfs` crate：它在 `no_std` 下必须开 `core_io` 特性，而该特性只支持
//! nightly。本项目用稳定 rustc，所以这里手写一份只读 FAT 解析——够读根目录里的图片文件，
//! 且完全可控。
//!
//! 支持：FAT16 / FAT32、按簇链读文件、8.3 短名匹配。不支持：LFN 长文件名、写、FAT12。

use alloc::vec::Vec;

use crate::sd::BlockDevice;

/// FAT 层错误。
#[derive(Debug, Clone, Copy)]
pub enum FatError {
    /// 底层块设备（SD 卡）错误。
    BlockDevice,
    /// 不是可识别的 FAT16/FAT32 卷（或 FAT12，暂不支持）。
    Unsupported,
    /// 文件没找到。
    NotFound,
}

/// FAT 子类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FatType {
    Fat16,
    Fat32,
}

/// 根目录里一个文件/目录项的精简视图。
#[derive(Debug, Clone, Copy)]
pub struct FileInfo {
    /// 11 字节短名（8 文件名 + 3 扩展名，空格填充，大写）。
    pub name: [u8; 11],
    /// 文件大小（字节）。
    pub size: u32,
    /// 首簇号。
    pub first_cluster: u32,
    /// 是否为目录。
    pub is_dir: bool,
}

/// 已挂载的 FAT 卷。
pub struct FatFs<B: BlockDevice> {
    bdev: B,
    fat_type: FatType,
    fat_start: u32,
    data_start: u32,
    sectors_per_cluster: u32,
    /// FAT16 的根目录起始扇区（FAT32 用不到）。
    root_dir_start: u32,
    /// FAT16 根目录扇区数（FAT32 用不到）。
    root_dir_sectors: u32,
    /// FAT32 根目录首簇（FAT16 用不到）。
    root_cluster: u32,
}

impl<B: BlockDevice> FatFs<B> {
    /// 读 SD 卡 0 扇区的 BPB，挂载 FAT 卷。
    pub fn mount(mut bdev: B) -> Result<Self, FatError> {
        let mut boot = [0u8; 512];
        bdev
            .read_blocks(&mut boot, 0, 1)
            .map_err(|_| FatError::BlockDevice)?;

        let bytes_per_sector = u16::from_le_bytes([boot[11], boot[12]]) as u32;
        if bytes_per_sector != 512 {
            return Err(FatError::Unsupported);
        }
        let sectors_per_cluster = boot[13] as u32;
        let reserved = u16::from_le_bytes([boot[14], boot[15]]) as u32;
        let num_fats = boot[16] as u32;
        let root_entry_count = u16::from_le_bytes([boot[17], boot[18]]) as u32;
        let total_16 = u16::from_le_bytes([boot[19], boot[20]]) as u32;
        let fat_size_16 = u16::from_le_bytes([boot[22], boot[23]]) as u32;
        let total_32 = u32::from_le_bytes([boot[32], boot[33], boot[34], boot[35]]);
        let fat_size_32 = u32::from_le_bytes([boot[36], boot[37], boot[38], boot[39]]);
        let root_cluster = u32::from_le_bytes([boot[44], boot[45], boot[46], boot[47]]);

        let fat_size = if fat_size_16 != 0 {
            fat_size_16
        } else {
            fat_size_32
        };
        let total_sectors = if total_16 != 0 { total_16 } else { total_32 };

        let fat_start = reserved;
        let root_dir_sectors =
            ((root_entry_count * 32) + (bytes_per_sector - 1)) / bytes_per_sector;
        let root_dir_start = reserved + num_fats * fat_size;
        // FAT16 的簇区（cluster 2）在根目录之后；FAT32 根目录就是簇链，簇区从 reserved+num_fats*fat_size 开始。
        let data_start = if root_entry_count != 0 {
            root_dir_start + root_dir_sectors
        } else {
            reserved + num_fats * fat_size
        };

        // 按数据簇数判断 FAT 类型。
        let data_sectors = total_sectors.saturating_sub(data_start);
        let clusters = data_sectors / sectors_per_cluster;
        let fat_type = if clusters < 4085 {
            return Err(FatError::Unsupported); // FAT12 暂不支持
        } else if clusters < 65525 {
            FatType::Fat16
        } else {
            FatType::Fat32
        };

        Ok(Self {
            bdev,
            fat_type,
            fat_start,
            data_start,
            sectors_per_cluster,
            root_dir_start,
            root_dir_sectors,
            root_cluster,
        })
    }

    /// 列出根目录里的所有文件/目录项。
    pub fn root_entries(&mut self) -> Result<Vec<FileInfo>, FatError> {
        let mut out = Vec::new();
        match self.fat_type {
            FatType::Fat16 => {
                for s in 0..self.root_dir_sectors {
                    self.scan_dir_sector(self.root_dir_start + s, &mut out)?;
                }
            }
            FatType::Fat32 => {
                let mut cluster = self.root_cluster;
                while cluster >= 2 && cluster < 0x0FFF_FFF8 {
                    self.scan_dir_cluster(cluster, &mut out)?;
                    cluster = self.next_cluster(cluster)?;
                }
            }
        }
        Ok(out)
    }

    /// 按 8.3 短名（如 `b"N0      PNG"`）找根目录里的文件。
    pub fn open(&mut self, name_8_3: &[u8; 11]) -> Result<FileInfo, FatError> {
        let entries = self.root_entries()?;
        for e in &entries {
            if !e.is_dir && e.name == *name_8_3 {
                return Ok(*e);
            }
        }
        Err(FatError::NotFound)
    }

    /// 把一个文件完整读进内存（按簇链）。返回的 `Vec` 落在 SDRAM 堆上。
    pub fn read_file(&mut self, info: &FileInfo) -> Result<Vec<u8>, FatError> {
        let mut out: Vec<u8> = Vec::new();
        let mut remaining = info.size;
        let mut cluster = info.first_cluster;
        while cluster >= 2 && cluster < 0x0FFF_FFF8 && remaining > 0 {
            self.read_cluster(cluster, &mut out, &mut remaining)?;
            cluster = self.next_cluster(cluster)?;
        }
        Ok(out)
    }

    // ---- 内部工具 ----

    fn scan_dir_sector(&mut self, sector: u32, out: &mut Vec<FileInfo>) -> Result<(), FatError> {
        let mut buf = [0u8; 512];
        self.bdev
            .read_blocks(&mut buf, sector, 1)
            .map_err(|_| FatError::BlockDevice)?;
        self.scan_dir_buffer(&buf, out);
        Ok(())
    }

    fn scan_dir_cluster(&mut self, cluster: u32, out: &mut Vec<FileInfo>) -> Result<(), FatError> {
        let first = self.data_start + (cluster - 2) * self.sectors_per_cluster;
        for s in 0..self.sectors_per_cluster {
            self.scan_dir_sector(first + s, out)?;
        }
        Ok(())
    }

    fn scan_dir_buffer(&self, buf: &[u8; 512], out: &mut Vec<FileInfo>) {
        let mut off = 0usize;
        while off + 32 <= 512 {
            let e = &buf[off..off + 32];
            let first = e[0];
            if first == 0x00 {
                break; // 目录结束
            }
            if first == 0xE5 || (e[11] & 0x0F) == 0x0F {
                off += 32; // 已删除 / LFN
                continue;
            }
            let is_dir = (e[11] & 0x10) != 0;
            let is_volume = (e[11] & 0x08) != 0;
            if is_volume {
                off += 32;
                continue;
            }
            let mut name = [0u8; 11];
            name.copy_from_slice(&e[0..11]);
            let size = u32::from_le_bytes([e[28], e[29], e[30], e[31]]);
            let lo = u16::from_le_bytes([e[26], e[27]]) as u32;
            let hi = u16::from_le_bytes([e[20], e[21]]) as u32;
            let first_cluster = (hi << 16) | lo;
            out.push(FileInfo {
                name,
                size,
                first_cluster,
                is_dir,
            });
            off += 32;
        }
    }

    fn read_cluster(
        &mut self,
        cluster: u32,
        out: &mut Vec<u8>,
        remaining: &mut u32,
    ) -> Result<(), FatError> {
        let first = self.data_start + (cluster - 2) * self.sectors_per_cluster;
        let mut buf = [0u8; 512];
        for _ in 0..self.sectors_per_cluster {
            if *remaining == 0 {
                break;
            }
            self.bdev
                .read_blocks(&mut buf, first, 1)
                .map_err(|_| FatError::BlockDevice)?;
            let take = (*remaining as usize).min(512);
            out.extend_from_slice(&buf[..take]);
            *remaining -= take as u32;
        }
        Ok(())
    }

    /// 查 FAT 表得到 `cluster` 的下一个簇号（FAT16 / FAT32 区分）。
    fn next_cluster(&mut self, cluster: u32) -> Result<u32, FatError> {
        let (entry_size, mask, eoc) = match self.fat_type {
            FatType::Fat16 => (2u32, 0x0000_FFFF, 0x0000_FFF8),
            FatType::Fat32 => (4u32, 0x0FFF_FFFF, 0x0FFF_FFF8),
        };
        let fat_offset = cluster * entry_size;
        let sector = self.fat_start + fat_offset / 512;
        let mut buf = [0u8; 512];
        self.bdev
            .read_blocks(&mut buf, sector, 1)
            .map_err(|_| FatError::BlockDevice)?;
        let off = (fat_offset % 512) as usize;
        let val = if entry_size == 2 {
            u32::from_le_bytes([buf[off], buf[off + 1], 0, 0])
        } else {
            u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]) & mask
        };
        if val >= eoc {
            Ok(0x0FFF_FFF8) // 用 EOC 哨兵表示结束
        } else {
            Ok(val)
        }
    }
}

/// 把一个 `"N0.PNG"` 这样的名字转成 11 字节 8.3 短名（大写、空格填充）。
///
/// 返回 `None` 表示名字不合法（超过 8.3 或含非法字符）。
pub fn to_8_3(name: &str) -> Option<[u8; 11]> {
    let mut out = [b' '; 11];
    let mut dot = name.find('.')?;
    if dot == 0 || dot > 8 {
        // 无扩展名或主名过长：当作全主名处理
        dot = name.len().min(8);
    }
    let main = &name[..dot];
    let ext = if dot < name.len() {
        &name[dot + 1..]
    } else {
        ""
    };
    if main.len() > 8 || ext.len() > 3 {
        return None;
    }
    for (i, c) in main.bytes().enumerate() {
        out[i] = c.to_ascii_uppercase();
    }
    for (i, c) in ext.bytes().enumerate() {
        out[8 + i] = c.to_ascii_uppercase();
    }
    Some(out)
}

/// 把 11 字节短名还原成可读字符串（`N0.PNG`），用于 RTT 打印。
pub fn from_8_3(name: &[u8; 11]) -> Vec<u8> {
    let mut s = Vec::new();
    for &c in &name[..8] {
        if c != b' ' {
            s.push(c);
        }
    }
    s.push(b'.');
    for &c in &name[8..11] {
        if c != b' ' {
            s.push(c);
        }
    }
    if s.last() == Some(&b'.') {
        s.pop();
    }
    s
}
