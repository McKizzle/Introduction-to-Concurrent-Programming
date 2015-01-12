#!/usr/bin/env bash

sudo init 3 # disable x11 in CentOS 
sudo sed -i '/id:[0-6]:initdefault:/c\id:3:initdefault:' /etc/inittab # set the default runlevel

# Disable Nouvau drivers
is_nouveau=$(lsmod | grep nouveau -c)
if [ $is_nouveau -ge 1 ]; then 
    echo "Nouveau drivers discovered. I will need to disable them."
    
    # Blacklist the nouveau driver by adding a rule to /etc/modprobe.d/disable-nouveau.conf
    echo "Then blacklist nouveau in /etc/modprobe.d/"
    NVIDIA_BLKLST=/etc/modprobe.d/disable-nouveau.conf
    sudo rm -f $NVIDIA_BLKLST
    sudo touch $NVIDIA_BLKLST
    echo "blacklist nouveau"         | sudo tee -a $NVIDIA_BLKLST # append to the file
    echo "options nouveau modeset=0" | sudo tee -a $NVIDIA_BLKLST # append to the file
    
    echo "First backup the current initramfs file and create a new one."
    sudo mv /boot/initramfs-$(uname -r).img /boot/initramfs-$(uname -r)-nouveau.img
    sudo dracut /boot/initramfs-$(uname -r).img $(uname -r)
    
    echo "Reboot the machine and rerun this script..."
    exit
else 
    echo "Safe to continue installation..."
fi

# Setup and install dkms. Allows the drivers to automatically recompile if 
# there are system updates. 
wget --no-clobber http://linux.dell.com/dkms/permalink/dkms-2.2.0.3-1.noarch.rpm
sudo rpm --install dkms-2.2.0.3-1.noarch.rpm
sudo yum clean expire-cache
sudo yum update
sudo yum install dkms

# Install the cuda libraries.
CUDAT_PREFIX=/opt/cuda-6.0
CUDAS_PREFIX=/opt/cuda-6.0-samples
TMP_CUDA=$(pwd)/
wget --no-clobber http://developer.download.nvidia.com/compute/cuda/6_0/rel/installers/cuda_6.0.37_linux_64.run
sudo sh cuda_6.0.37_linux_64.run -toolkit -toolkitpath=$CUDAT_PREFIX \
	-samples -samplespath=$CUDAS_PREFIX \
	-silent	-tmpdir $TMP_CUDA \
	-driver
nvidia-persistenced --persistence-mode #keep the driver turned on all the time 'speeds up exectution' 

# so that the user can run the nvcc command 
echo "export PATH=$CUDAT_PREFIX/bin:\$PATH" | sudo tee -a /etc/bashrc # append to the file

# ldconfig will handle this part now. 
#echo "export LD_LBRARY_PATH=$CUDAT_PREFIX/lib64:\$LD_LBRARY_PATH" | sudo tee -a /etc/bashrc

# so that libraries are exported 
CUDA6_CONF=/etc/ld.so.conf.d/cuda6.0.conf
sudo touch $CUDA6_CONF 
echo "$CUDAT_PREFIX/lib64" | sudo tee -a $CUDA6_CONF # append to the file
echo "$CUDAT_PREFIX/lib" | sudo tee -a $CUDA6_CONF # append to the file
sudo ldconfig # update the library linker. 

