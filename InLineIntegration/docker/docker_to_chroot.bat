@echo off
rem This script takes a Docker image and creates a chroot image (.img)
rem Note that this script also requires docker_tar_to_chroot.sh to be located in the same folder

rem Syntax: docker_to_chroot.bat kspacekelvin/fire-python fire-python-chroot.img

set DOCKER_NAME=%1
set CHROOT_FILE=%2
set EXPORT_FILE=docker-export.tar

rem Create a Docker container and export to a .tar file
echo ------------------------------------------------------------
echo Exporting Docker image %DOCKER_NAME%
echo ------------------------------------------------------------

docker create --name tmpimage %DOCKER_NAME%
docker export -o %EXPORT_FILE% tmpimage
docker rm tmpimage

rem Run a privileged Docker to create the chroot file
docker run -it --rm          ^
           --privileged=true ^
           -v %cd%:/share    ^
           ubuntu            ^
           /bin/bash /share/docker_tar_to_chroot.sh /share/%EXPORT_FILE% /share/%CHROOT_FILE%

del %EXPORT_FILE%
