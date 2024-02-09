#!/bin/bash


bash scripts/PSM.sh > logs/PSM/PSM.log 2>&1;
bash scripts/PSM_full.sh > logs/PSM/SMAP_PSM.log 2>&1;
bash scripts/SWAT.sh > logs/SWAT/SWAT.log 2>&1;
bash scripts/SWAT_full.sh > logs/SWAT/SWAT_full.log 2>&1