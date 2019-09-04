---
layout:     post
title:      "How to setup a deep learning machine"
subtitle:   "OS and Cuda Installation"
date:       2014-06-14
author:     "Shen Xu"
image: "img/cuda_img.jpg"
published: true
hide-in-home: false
markup: "mmark"
tags:
    - TECH
categories: [ TECH ]    
---
---
title: How to Setup a Deep Learning Machine
layout: default
---
# How to setup a Deep Learning Machine

I spent sometime in past few days to setup a new OS and necessary softwares for deep learning since I got a new GPU (GTX 980).

I found out it is hard to setup a system running deep learning envrionment, especially the CUDA part. A bunch of tricks would soon be forgotten if not written down. So I wrote it down this time and hopefully it would serve as a reminder for myself and make sense to everyone. (Although I'm pretty sure by the next time I need to probe this, things change.)

## Install Operting System

I just installed a Ubuntu 14.04 this time, because seems it is the most stable version until now. The installation is quite straight forward, download the image from [Utunbu website](http://www.ubuntu.com/download/desktop) and copy the image to USB drive and install it on the SSD.

## Install Nvidia Driver and Cuda

This is the most import step. When I search around the web, a lot people suggest using

```bash
get-apt install nvidia-recent
```
However, it does __NOT__ work for me and a lot other people I belive. After I installed this driver from Ubuntu repositories, I could not login anymore (the screen would be freeze there). There are two reasons:

 - The nvidia-recent version is too old. I think the Ubuntu repostiry has not been updated for a while.
 - The installed driver will overwrite some GL files, which would cause a lot problem for screen display.

Since we will install CUDA later and CUDA package includes the Nvidia driver, we can just download the CUDA driver and install the driver includes in the CUDA.

I just download the newest [CUDA 7](https://developer.nvidia.com/cuda-downloads). The file I used is ".run".

Without installing any driver, Ubuntu uses a default driver for Nvidia cards called "nouveau driver". Before we install the correct driver (from Nvidia), we need to turn this "nouveau" driver off. open the file "/etc/modprobe.d/blacklist.conf",append the following at the end of the "conf" file:
```
blacklist nouveau
options nouveau modeset=0
```

After this, we can install the Nvidia driver now. As we know, the Nvidia driver takes care of all graph display; so the "X" server and Ubuntu GUI needs to be closed. We can reboot the computer, in the login page, do NOT login, click "CTRL-ALT-F1" to drop to the terminal without GUI. Then turn off X server by:
```bash
sudo service lightdm stop
```

I also find out it is easy to operate this and the following by sudo as super user.
```bash
sudo su
```

Then extract "nvidia-driver", "cuda", and "cuda-examples".

 - Step 1:
```bash
./<cuda-archive-name> --extract=<your_fav_dir>
```


 - Step 2: Navigate to your_fav_dir, change all three extracted .run files to executable.
```bash
cd <your_fav_dir>
sudo chmod +x NVIDIA-Linux-x86_64-346.46.run
sudo chmod +x cuda-linux64-rel-7.0.28-19326674.run
sudo chmod +x cuda-samples-linux-7.0.28-19326674.run
```


 - Step 3:  Install the Nvidia driver by (__this is the most important step__)
```bash
sudo ./<NVIDIA_driver_run_name> --no-opengl-files
```
The --no-opengl-files option prevents overwriting of some GL files. If you don't pass this your screen will freeze after login. Also, select 'no' when it asks you update the xorg.conf file.


 - Step 4: Reboot to normal, now you will see the screen resolution becomes good rightaway. You can verify through the Additional Drivers utility that you are using a manually installed driver.


 - Step 5: Navigate again to your_fav_dir and install the cuda toolkit and samples in pretty much the same manner (you don't need to pass any special options now).

 __Enjoy CUDA !!__


## Install Theano

## Install Torch 7
