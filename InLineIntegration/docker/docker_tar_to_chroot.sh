#!/bin/bash
# This script takes a Docker container export (.tar) creates a chroot image (.img)
# Note that root privileges are required to mount the loopback images 

# Syntax: ./docker_tar_to_chroot.sh docker-export.tar chroot.img

EXPORT_FILE=${1}
CHROOT_FILE=${2}

exportSize=$(stat --format=%s "${EXPORT_FILE}")

# Add a minimum buffer of 125 MB free space to account for filesystem overhead
chrootMinSize=$(( exportSize/(1024*1024) + 125 ))

# Round up to the nearest 50 MB
chrootSize=$(( ((${chrootMinSize%.*})/50+1)*50 ))

echo ------------------------------------------------------------
echo Docker export file is $(( exportSize/(1024*1024) )) MB
echo Creating chroot file ${CHROOT_FILE} of size $chrootSize MB
echo ------------------------------------------------------------

if test -f "${CHROOT_FILE}"; then
    echo "Warning -- ${CHROOT_FILE} exists and will be overwritten!"
    rm ${CHROOT_FILE}
fi

# Create blank ext3 chroot image
dd if=/dev/zero of=${CHROOT_FILE} bs=1M count=${chrootSize}
mke2fs -F -t ext3 ${CHROOT_FILE}

# Mount image and copy contents from tar export
mkdir /mnt/chroot
mount -o loop ${CHROOT_FILE} /mnt/chroot
tar -xf ${EXPORT_FILE} --directory=/mnt/chroot --totals

# Show the amount of free space left on the chroot
df -h

umount /mnt/chroot
