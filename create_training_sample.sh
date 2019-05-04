#!/bin/bash
DATADIR=mnist
mkdir $DATADIR/small_training

mkdir $DATADIR/small_training/0
mkdir $DATADIR/small_training/1
mkdir $DATADIR/small_training/2
mkdir $DATADIR/small_training/3
mkdir $DATADIR/small_training/4
mkdir $DATADIR/small_training/5
mkdir $DATADIR/small_training/6
mkdir $DATADIR/small_training/7
mkdir $DATADIR/small_training/8
mkdir $DATADIR/small_training/9

cp $DATADIR/training/0/0000??.png $DATADIR/small_training/0/
cp $DATADIR/training/1/0000??.png $DATADIR/small_training/1/
cp $DATADIR/training/2/0000??.png $DATADIR/small_training/2/
cp $DATADIR/training/3/0000??.png $DATADIR/small_training/3/
cp $DATADIR/training/4/0000??.png $DATADIR/small_training/4/
cp $DATADIR/training/5/0000??.png $DATADIR/small_training/5/
cp $DATADIR/training/6/0000??.png $DATADIR/small_training/6/
cp $DATADIR/training/7/0000??.png $DATADIR/small_training/7/
cp $DATADIR/training/8/0000??.png $DATADIR/small_training/8/
cp $DATADIR/training/9/0000??.png $DATADIR/small_training/9/

#cp $DATADIR/training/0/0001??.png $DATADIR/small_training/0/
#cp $DATADIR/training/1/0001??.png $DATADIR/small_training/1/
#cp $DATADIR/training/2/0001??.png $DATADIR/small_training/2/
#cp $DATADIR/training/3/0001??.png $DATADIR/small_training/3/
#cp $DATADIR/training/4/0001??.png $DATADIR/small_training/4/
#cp $DATADIR/training/5/0001??.png $DATADIR/small_training/5/
#cp $DATADIR/training/6/0001??.png $DATADIR/small_training/6/
#cp $DATADIR/training/7/0001??.png $DATADIR/small_training/7/
#cp $DATADIR/training/8/0001??.png $DATADIR/small_training/8/
#cp $DATADIR/training/9/0001??.png $DATADIR/small_training/9/