#!/usr/bin/env python
# -*- coding: utf-8 -*-
from recognition_lib import *
import argparse

 
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--camera", action="store_true")
args = ap.parse_args()

if args.camera:
	recognition_camera()
else:
	recognition_images()
