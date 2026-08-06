#!/usr/bin/env python3
"""Rename SONAME and DT_NEEDED entries of an ELF shared library in place.

The V100 driver stack must load next to the system CUDA stack in the same
process. Both stacks carry identical SONAMEs, so ld.so would reuse a single
copy. Renaming "libcu*" to "libcv*" keeps the byte length identical, so the
patch is a pure byte overwrite: section sizes, offsets, symbol tables and
.gnu.hash all stay valid.

patchelf cannot be used here - it rewrites the whole ELF and corrupts
libcublasLt.so.12 (749 MB).
"""

import shutil
import struct
import sys

DT_NEEDED    = 1
DT_STRTAB    = 5
DT_STRSZ     = 10
DT_SONAME    = 14
DT_VERDEF    = 0x6ffffffc
DT_VERDEFNUM = 0x6ffffffd
DT_VERNEED   = 0x6ffffffe
DT_VERNEEDNUM= 0x6fffffff


def elf_hash(name):
    h = 0
    for c in name.encode():
        h = (h << 4) + c
        g = h & 0xf0000000
        if g:
            h ^= g >> 24
        h &= ~g & 0xffffffff
    return h

# Only these names are renamed. Anything else (libc, libm, ...) must stay shared.
RENAME = {
    "libcuda.so.1":      "libcvda.so.1",
    "libcudart.so.12":   "libcvdart.so.12",
    "libcublas.so.12":   "libcvblas.so.12",
    "libcublasLt.so.12": "libcvblasLt.so.12",
}


def vaddr_to_off(loads, vaddr):
    for p_off, p_vaddr, p_filesz in loads:
        if p_vaddr <= vaddr < p_vaddr + p_filesz:
            return p_off + (vaddr - p_vaddr)
    raise ValueError("vaddr 0x%x not inside any PT_LOAD" % vaddr)


def patch_rodata(path):
    """Rename library names that are dlopen'ed at run time, not in DT_NEEDED.

    libcudart and libcublas reach the driver with dlopen("libcuda.so.1"), so the
    name lives in .rodata. Same-length rename keeps every offset valid.
    """
    with open(path, "r+b") as f:
        blob = f.read()
        changed = []
        for old, new in RENAME.items():
            needle = old.encode() + b"\0"
            pos = blob.find(needle)
            n = 0
            while pos != -1:
                f.seek(pos)
                f.write(new.encode())
                n += 1
                pos = blob.find(needle, pos + 1)
            if n:
                changed.append("rodata %s -> %s (x%d)" % (old, new, n))
    return changed


def patch(path):
    with open(path, "r+b") as f:
        data = bytearray(f.read(64))
        if data[:4] != b"\x7fELF" or data[4] != 2:
            raise ValueError("not a 64-bit ELF: " + path)

        e_phoff, = struct.unpack_from("<Q", data, 0x20)
        e_phentsize, e_phnum = struct.unpack_from("<HH", data, 0x36)

        f.seek(e_phoff)
        phdrs = f.read(e_phentsize * e_phnum)

        loads = []
        dyn = None
        for i in range(e_phnum):
            p = phdrs[i * e_phentsize:(i + 1) * e_phentsize]
            p_type, = struct.unpack_from("<I", p, 0)
            p_offset, p_vaddr = struct.unpack_from("<QQ", p, 0x08)
            p_filesz, = struct.unpack_from("<Q", p, 0x20)
            if p_type == 1:      # PT_LOAD
                loads.append((p_offset, p_vaddr, p_filesz))
            elif p_type == 2:    # PT_DYNAMIC
                dyn = (p_offset, p_filesz)

        if dyn is None:
            raise ValueError("no PT_DYNAMIC: " + path)

        f.seek(dyn[0])
        dynamic = f.read(dyn[1])

        strtab_vaddr = None
        strsz = None
        verdef_vaddr = verdef_num = None
        verneed_vaddr = verneed_num = None
        for off in range(0, len(dynamic), 16):
            d_tag, d_val = struct.unpack_from("<qQ", dynamic, off)
            if d_tag == 0:
                break
            if d_tag == DT_STRTAB:
                strtab_vaddr = d_val
            elif d_tag == DT_STRSZ:
                strsz = d_val
            elif d_tag == DT_VERDEF:
                verdef_vaddr = d_val
            elif d_tag == DT_VERDEFNUM:
                verdef_num = d_val
            elif d_tag == DT_VERNEED:
                verneed_vaddr = d_val
            elif d_tag == DT_VERNEEDNUM:
                verneed_num = d_val

        if strtab_vaddr is None or strsz is None:
            raise ValueError("no DT_STRTAB/DT_STRSZ: " + path)
        strtab_off = vaddr_to_off(loads, strtab_vaddr)

        # Rewrite every matching name in .dynstr, not just DT_SONAME/DT_NEEDED.
        # The same strings name the symbol versions in .gnu.version_d; leaving
        # those alone makes the linker ask for a version the library lacks.
        # .dynstr holds no hashes, so equal-length edits stay consistent.
        f.seek(strtab_off)
        strtab = bytearray(f.read(strsz))

        changed = []
        for old, new in RENAME.items():
            n = 0
            start = 0
            needle = b"\0" + old.encode() + b"\0"
            while True:
                pos = strtab.find(needle, start)
                if pos == -1:
                    break
                strtab[pos + 1:pos + 1 + len(old)] = new.encode()
                n += 1
                start = pos + 1
            if n:
                changed.append("dynstr %s -> %s (x%d)" % (old, new, n))

        f.seek(strtab_off)
        f.write(strtab)

        def name_at(offset):
            end = strtab.find(b"\0", offset)
            return strtab[offset:end].decode()

        # Verdef/Verneed carry an ELF hash of the version name. Renaming the
        # string leaves the hash stale, and ld.so looks versions up by hash.
        if verdef_vaddr and verdef_num:
            off = vaddr_to_off(loads, verdef_vaddr)
            for _ in range(verdef_num):
                f.seek(off)
                vd = f.read(20)
                vd_hash, vd_aux, vd_next = struct.unpack_from("<III", vd, 8)
                f.seek(off + vd_aux)
                vda_name, = struct.unpack("<I", f.read(4))
                new_hash = elf_hash(name_at(vda_name))
                if new_hash != vd_hash:
                    f.seek(off + 8)
                    f.write(struct.pack("<I", new_hash))
                    changed.append("verdef hash %s" % name_at(vda_name))
                if not vd_next:
                    break
                off += vd_next

        if verneed_vaddr and verneed_num:
            off = vaddr_to_off(loads, verneed_vaddr)
            for _ in range(verneed_num):
                f.seek(off)
                vn = f.read(16)
                vn_cnt, = struct.unpack_from("<H", vn, 2)
                vn_aux, vn_next = struct.unpack_from("<II", vn, 8)
                aux = off + vn_aux
                for _ in range(vn_cnt):
                    f.seek(aux)
                    vna = f.read(16)
                    vna_hash, = struct.unpack_from("<I", vna, 0)
                    vna_name, vna_next = struct.unpack_from("<II", vna, 8)
                    new_hash = elf_hash(name_at(vna_name))
                    if new_hash != vna_hash:
                        f.seek(aux)
                        f.write(struct.pack("<I", new_hash))
                        changed.append("verneed hash %s" % name_at(vna_name))
                    if not vna_next:
                        break
                    aux += vna_next
                if not vn_next:
                    break
                off += vn_next

    return changed


def main():
    if len(sys.argv) != 3:
        print("usage: rename_soname.py <src.so> <dst.so>", file=sys.stderr)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    if src != dst:
        shutil.copyfile(src, dst)
    for line in patch(dst) + patch_rodata(dst):
        print("  %s" % line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
