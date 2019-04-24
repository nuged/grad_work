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

cp $DATADIR/training/0/000???.png $DATADIR/small_training/0/
cp $DATADIR/training/1/000???.png $DATADIR/small_training/1/
cp $DATADIR/training/2/000???.png $DATADIR/small_training/2/
cp $DATADIR/training/3/000???.png $DATADIR/small_training/3/
cp $DATADIR/training/4/000???.png $DATADIR/small_training/4/
cp $DATADIR/training/5/000???.png $DATADIR/small_training/5/
cp $DATADIR/training/6/000???.png $DATADIR/small_training/6/
cp $DATADIR/training/7/000???.png $DATADIR/small_training/7/
cp $DATADIR/training/8/000???.png $DATADIR/small_training/8/
cp $DATADIR/training/9/000???.png $DATADIR/small_training/9/

#cp $DATADIR/training/0/001???.png $DATADIR/small_training/0/
#cp $DATADIR/training/1/001???.png $DATADIR/small_training/1/
#cp $DATADIR/training/2/001???.png $DATADIR/small_training/2/
#cp $DATADIR/training/3/001???.png $DATADIR/small_training/3/
#cp $DATADIR/training/4/001???.png $DATADIR/small_training/4/
#cp $DATADIR/training/5/001???.png $DATADIR/small_training/5/
#cp $DATADIR/training/6/001???.png $DATADIR/small_training/6/
#cp $DATADIR/training/7/001???.png $DATADIR/small_training/7/
#cp $DATADIR/training/8/001???.png $DATADIR/small_training/8/
#cp $DATADIR/training/9/001???.png $DATADIR/small_training/9/