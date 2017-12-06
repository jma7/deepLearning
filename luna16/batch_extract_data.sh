#!/bin/bash
path="/work/05268/junma7/maverick/luna16/"
tempFile="check.zip"
for file in subset2.zip subset3.zip subset4.zip subset5.zip subset6.zip subset7.zip subset8.zip subset9.zip
	do 
		zip -FFv $path$file --out $path$tempFile
		unzip $path$tempFile -d $path
	done
		
